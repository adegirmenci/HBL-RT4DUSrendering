#ifndef VOLUMERENDER_H
#define VOLUMERENDER_H

#include <QObject>
#include <QCoreApplication>

// OpenGL Graphics includes
#include <GL/glew.h>
#include <GL/freeglut.h>

// CUDA Runtime, Interop, and includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

// CUDA utilities
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

// Helper functions
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_timer.h>

typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned char VolumeType;

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f

extern "C"
void setTextureFilterMode(bool bLinearFilter);

extern "C"
void initCuda(void **h_volume, cudaExtent volumeSize);

extern "C"
void freeCudaBuffers();

extern "C"
void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH,
                              float density, float brightness, float transferOffset, float transferScale, float lowerThresh, int currFrame);
extern "C"
void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix);

void initPixelBuffer();
void computeFPS();
void render();

void idle();
void keyboard(unsigned char key, int x, int y);
void display();
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void reshape(int w, int h);
void cleanup();

int iDivUp(int a, int b);
int chooseCudaDevice(bool bUseOpenGL);

void initGL();
void initialize();

unsigned int m_nTimeFrames = 1;
unsigned int m_currFrameIdx = 0;
unsigned int m_frameGaps = 8;
unsigned int m_framesShown = 0;

QString m_volumeFilename;
cudaExtent m_volumeSize;

uint m_width = 1024, m_height = 1024;
dim3 m_blockSize = dim3(32,32);
dim3 m_gridSize;

float3 m_viewRotation;
float3 m_viewTranslation;
float m_invViewMatrix[12];

float m_density;
float m_brightness;
float m_transferOffset;
float m_transferScale;
float m_lowerThresh;
bool m_linearFiltering = true;

GLuint m_pbo = 0; // OpenGL pixel buffer object
GLuint* m_texArray = NULL;    // OpenGL texture object
struct cudaGraphicsResource *m_cuda_pbo_resource;  // CUDA Graphics Resource (to transfer PBO)

StopWatchInterface *m_timer = 0;
int m_frameCheckNumber = 2;
int m_fpsCount = 0;        // FPS count for averaging
int m_fpsLimit = 0;        // FPS limit for sampling
int m_g_Index = 0;
unsigned int m_frameCount = 0;

int m_ox = 0, m_oy = 0;
int m_buttonState = 0;


std::vector<void *> loadRawFile(char *filename, size_t size);


void initialize()
{
    //start logs
    printf("%s Starting Volume Renderer\n\n");

    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    initGL();

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    chooseCudaDevice(true);

    char *filename = "volume_0015150725_031503_1.raw";
    m_volumeSize = make_cudaExtent(245,209,376);

    // load volume data
    char *path = sdkFindFilePath(filename, "C:\\ProgramData\\NVIDIA Corporation\\CUDA Samples\\v7.0\\2_Graphics\\volumeRender\\data\\study15_15vols");

    if (path == 0)
    {
     printf("Error finding file '%s'\n", filename);
     exit(EXIT_FAILURE);
    }

    size_t size = m_volumeSize.width*m_volumeSize.height*m_volumeSize.depth*sizeof(VolumeType);

    std::vector<void *> h_volume = loadRawFile(path, size);

    printf("All data loaded.\n");

    initCuda(&h_volume[0], m_volumeSize);
    printf("CUDA init success.\n");

    sdkCreateTimer(&m_timer);

    // calculate new grid size
    m_gridSize = dim3(iDivUp(m_width, m_blockSize.x), iDivUp(m_height, m_blockSize.y));

    // This is the normal rendering path for VolumeRender
    glutDisplayFunc(&display);
    glutKeyboardFunc(&keyboard);
    glutMouseFunc(&mouse);
    glutMotionFunc(&motion);
    glutReshapeFunc(&reshape);
    glutIdleFunc(&idle);

    initPixelBuffer();

    glutCloseFunc(cleanup);

    glutMainLoop();
}

void computeFPS()
{
    m_frameCount++;
    m_fpsCount++;

    if (m_fpsCount == m_fpsLimit)
    {
        // char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&m_timer) / 1000.f);
        // sprintf(fps, "Volume Render: %3.1f fps, currFrame %d", ifps, currFrame);

        // glutSetWindowTitle(fps);
        m_fpsCount = 0;

        m_fpsLimit = (int)MAX(1.f, ifps);
        sdkResetTimer(&m_timer);
    }
    char fps[256];
    sprintf(fps, "currFrame %d", m_currFrameIdx);

    glutSetWindowTitle(fps);
}

// render image using CUDA
void render()
{
    copyInvViewMatrix(m_invViewMatrix, sizeof(float4)*3);

    // map PBO to get CUDA device pointer
    uint *d_output;
    // map PBO to get CUDA device pointer
    checkCudaErrors(cudaGraphicsMapResources(1, &m_cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,
                                                         m_cuda_pbo_resource));
    //printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

    // clear image
    checkCudaErrors(cudaMemset(d_output, 0, m_width*m_height*4));

    // call CUDA kernel, writing results to PBO
    render_kernel(m_gridSize, m_blockSize, d_output, m_width, m_height, m_density, m_brightness, m_transferOffset, m_transferScale, m_lowerThresh, m_currFrameIdx);

    getLastCudaError("kernel failed");

    checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cuda_pbo_resource, 0));
}

// display results using OpenGL (called by GLUT)
void display()
{
    // go to next time step
    m_framesShown++;
    if(m_framesShown == m_frameGaps)
    {
        m_framesShown = 0;
        m_currFrameIdx++;
        m_currFrameIdx = m_currFrameIdx % m_nTimeFrames;
    }

    sdkStartTimer(&m_timer);

    // use OpenGL to build view matrix
    GLfloat modelView[16];
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glRotatef(-m_viewRotation.x, 1.0, 0.0, 0.0);
    glRotatef(-m_viewRotation.y, 0.0, 1.0, 0.0);
    glRotatef(-m_viewRotation.z, 0.0, 0.0, 1.0);
    glTranslatef(-m_viewTranslation.x, -m_viewTranslation.y, -m_viewTranslation.z);
    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
    glPopMatrix();

    m_invViewMatrix[0] = modelView[0];
    m_invViewMatrix[1] = modelView[4];
    m_invViewMatrix[2] = modelView[8];
    m_invViewMatrix[3] = modelView[12];
    m_invViewMatrix[4] = modelView[1];
    m_invViewMatrix[5] = modelView[5];
    m_invViewMatrix[6] = modelView[9];
    m_invViewMatrix[7] = modelView[13];
    m_invViewMatrix[8] = modelView[2];
    m_invViewMatrix[9] = modelView[6];
    m_invViewMatrix[10] = modelView[10];
    m_invViewMatrix[11] = modelView[14];

    render();

    // display results
    glClear(GL_COLOR_BUFFER_BIT);

    // draw image from PBO
    glDisable(GL_DEPTH_TEST);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    // draw using texture

    // copy from pbo to texture
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, m_pbo);
    glActiveTextureARB(GL_TEXTURE0_ARB);
    glBindTexture(GL_TEXTURE_2D, m_texArray[0]);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    //glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, texArray[0]);

    // draw textured quad
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2f(0, 0);
    glTexCoord2f(1, 0);
    glVertex2f(1, 0);
    glTexCoord2f(1, 1);
    glVertex2f(1, 1);
    glTexCoord2f(0, 1);
    glVertex2f(0, 1);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);

    glutSwapBuffers();
    glutReportErrors();

    sdkStopTimer(&m_timer);

    computeFPS();

}

void idle()
{
    glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 27:
            glutDestroyWindow(glutGetWindow());
            return;
            break;

        case 'f':
            m_linearFiltering = !m_linearFiltering;
            setTextureFilterMode(m_linearFiltering);
            break;

        case '+':
            m_density += 0.01f;
            break;

        case '-':
            m_density -= 0.01f;
            break;

        case ']':
            m_brightness += 0.1f;
            break;

        case '[':
            m_brightness -= 0.1f;
            break;

        case ';':
            m_transferOffset += 0.01f;
            break;

        case '\'':
            m_transferOffset -= 0.01f;
            break;

        case '.':
            m_transferScale += 0.01f;
            break;

        case ',':
            m_transferScale -= 0.01f;
            break;

        case 'm':
            m_lowerThresh += 0.01f;
            break;

        case 'n':
            m_lowerThresh -= 0.01f;
            break;

        case 'a':
            m_viewRotation.z += 25 / 5.0f;
            break;

        case 'd':
            m_viewRotation.z -= 25 / 5.0f;
            break;

        case 'w':
            m_viewRotation.y += 25 / 5.0f;
            break;

        case 's':
            m_viewRotation.y -= 25 / 5.0f;
            break;

        case 'q':
            m_viewRotation.x += 25 / 5.0f;
            break;

        case 'e':
            m_viewRotation.x -= 25 / 5.0f;
            break;

        default:
            break;
    }

    printf("density = %.2f, brightness = %.2f, transferOffset = %.2f, transferScale = %.2f, lowerThresh = %.2f\n", m_density, m_brightness, m_transferOffset, m_transferScale, m_lowerThresh);
    glutPostRedisplay();
}

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        m_buttonState  |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        m_buttonState = 0;
    }

    m_ox = x;
    m_oy = y;

    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - m_ox);
    dy = (float)(y - m_oy);

    if (m_buttonState == 4)
    {
        // right = zoom
        m_viewTranslation.z += dy / 100.0f;
    }
    else if (m_buttonState == 2)
    {
        // middle = translate
        m_viewTranslation.x += dx / 100.0f;
        m_viewTranslation.y -= dy / 100.0f;
    }
    else if (m_buttonState == 1)
    {
        // left = rotate

        m_viewRotation.x += dy / 5.0f;
        m_viewRotation.y += dx / 5.0f;
    }

    m_ox = x;
    m_oy = y;

    glutPostRedisplay();
}

int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void reshape(int w, int h)
{
    m_width = w;
    m_height = h;
    initPixelBuffer();

    // calculate new grid size
    m_gridSize = dim3(iDivUp(m_width, m_blockSize.x), iDivUp(m_height, m_blockSize.y));

    glViewport(0, 0, w, h);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

void cleanup()
{
    sdkDeleteTimer(&m_timer);

    freeCudaBuffers();

    if (m_pbo)
    {
        cudaGraphicsUnregisterResource(m_cuda_pbo_resource);
        glDeleteBuffersARB(1, &m_pbo);
        glDeleteTextures(m_nTimeFrames, m_texArray);
    }
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    printf("Exiting program, resetting device.\n");

    cudaDeviceReset();
}

void initGL()
{
    // initialize GLUT callback functions
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(m_width, m_height);
    glutCreateWindow("CUDA volume rendering");

    glewInit();

    if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object"))
    {
        printf("Required OpenGL extensions missing.");
        exit(EXIT_SUCCESS);
    }
}

void initPixelBuffer()
{
    if (m_pbo)
    {
        // unregister this buffer object from CUDA C
        checkCudaErrors(cudaGraphicsUnregisterResource(m_cuda_pbo_resource));

        // delete old buffer
        glDeleteBuffersARB(1, &m_pbo);
        glDeleteTextures(m_nTimeFrames, m_texArray);
    }

    // create pixel buffer object for display
    glGenBuffersARB(1, &m_pbo);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, m_pbo);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, m_width*m_height*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_cuda_pbo_resource, m_pbo, cudaGraphicsMapFlagsWriteDiscard));

    m_texArray = new GLuint[m_nTimeFrames];

    // create texture for display
    glGenTextures(m_nTimeFrames, m_texArray);
    for(int i = 0; i < 1; i++)
    {
        glBindTexture(GL_TEXTURE_2D, m_texArray[i]); // use 0 instead of i ?
        // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

}

// Load raw data from disk
std::vector<void *> loadRawFile(char *filename, size_t size)
{
    std::vector<void *> data;// = malloc(nTimeFrames * sizeof(void *));

    for(int i = 1; i <= m_nTimeFrames; i++)
    {
        std::ostringstream s;
        s << i;

        std::string buffer(filename);
        buffer = buffer.substr(0, buffer.size()-5);
        buffer.append(s.str());
        buffer.append(".raw");

        FILE *fp = fopen(buffer.c_str(), "rb");
        //FILE *fp = fopen(filename, "rb");

        if (!fp)
        {
            fprintf(stderr, "Error opening file '%s'\n", buffer.c_str());
            return data;
        }

        void *dat_ = malloc(size);

        size_t read = fread(dat_, 1, size, fp);
        fclose(fp);

        data.push_back(dat_);

        printf("Read '%s', %d bytes\n", buffer.c_str(), read);
    }
    return data;
}

// General initialization call for CUDA Device
int chooseCudaDevice(bool bUseOpenGL)
{
    int result = 0;

    int argc = 1;
    QString a = QCoreApplication::arguments().at(0);
    const char *argv= a.toStdString().c_str();

    if (bUseOpenGL)
    {
        result = findCudaGLDevice(argc, &argv);
    }
    else
    {
        result = findCudaDevice(argc, &argv);
    }

    return result;
}


#endif // VOLUMERENDER_H
