#include "extdopenglwidget.h"

//ExtdOpenGLwidget::ExtdOpenGLwidget(QWidget* parent, Qt::WindowFlags f)
//    : QOpenGLWidget(parent, f)
ExtdOpenGLwidget::ExtdOpenGLwidget(QOpenGLWindow::UpdateBehavior updateBehavior, QWindow *parent)
    : QOpenGLWindow(updateBehavior, parent)
{
    m_texArray = NULL;
    m_pbo = 0;

    m_paintParams.nTimeFrames  = 1;
    m_paintParams.currFrameIdx = 0;
    m_paintParams.frameGaps    = 8;
    m_paintParams.framesShown  = 0;

    m_paintParams.width  = 512;
    m_paintParams.height = 512;
    setWidth(m_paintParams.width);
    setHeight(m_paintParams.height);

    m_cudaParams.blockSize = dim3(32,32);
    m_cudaParams.gridSize =  dim3(
                iDivUp(m_paintParams.width, m_cudaParams.blockSize.x),
                iDivUp(m_paintParams.height, m_cudaParams.blockSize.y));

    m_raytraceParams.linearFiltering =  true;
    m_raytraceParams.density         =  0.18f;
    m_raytraceParams.brightness      =  0.90f;
    m_raytraceParams.transferOffset  = -0.05f;
    m_raytraceParams.transferScale   =  1.25f;
    m_raytraceParams.lowerThresh     =  0.0f;

    m_fpsParams.timer = 0;
    m_fpsParams.frameCheckNumber = 2;
    m_fpsParams.fpsCount = 0;
    m_fpsParams.fpsLimit = 0;
    m_fpsParams.g_Index = 0;
    m_fpsParams.frameCount = 0;

    m_mouseParams.ox = 0;
    m_mouseParams.oy = 0;
    m_mouseParams.buttonState = 0;

    m_viewportParams.viewTranslation = make_float3(0.0, 0.0, -4.0f);

    printf("ExtdOpenGLWidget initialized.\n");

    sdkCreateTimer(&m_fpsParams.timer);

    show();
}


ExtdOpenGLwidget::~ExtdOpenGLwidget()
{


    std::cout << "BYEBYE" << std::endl;
}

void ExtdOpenGLwidget::initializeGL()
{
    makeCurrent();
    initializeOpenGLFunctions();
    //this->context()->create();

    emit sendMsgToGUI(tr("Init GL."));
    printf("Init GL\n");

    cudaDeviceProp deviceProp;
    int devID = 0;
    devID = gpuGetMaxGflopsDeviceId();
    checkCudaErrors(cudaGLSetGLDevice(devID));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
    emit sendMsgToGUI(tr("GPU Device %1: \"%2\" with compute capability %3.%4\n")
                         .arg(devID).arg(deviceProp.name).arg(deviceProp.major).arg(deviceProp.minor));
    //std::cout << "!" << std::endl;

//    int c = 1;
//    char dummy[] = "";
//    char *ptrrr = dummy;

//        glutInit( &c, &ptrrr );
//        glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
//        glutInitWindowSize(m_paintParams.width, m_paintParams.height);
//        glutCreateWindow("CUDA volume rendering");
//        glewInit();

//QOpenGLContext *ctx = this->context();
    connect(context(), &QOpenGLContext::aboutToBeDestroyed,
            this, &ExtdOpenGLwidget::cleanup);
}

void ExtdOpenGLwidget::resizeGL(int w, int h)
{
    m_paintParams.width = w;
    m_paintParams.height = h;
    initPixelBuffer();

    // calculate new grid size
    m_cudaParams.gridSize = dim3(
                iDivUp(w, m_cudaParams.blockSize.x),
                iDivUp(h, m_cudaParams.blockSize.y));

    glViewport(0, 0, w, h);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

void ExtdOpenGLwidget::paintGL()
{
    // go to next time step
    m_paintParams.framesShown++;
    if(m_paintParams.framesShown == m_paintParams.frameGaps)
    {
        m_paintParams.framesShown = 0;
        m_paintParams.currFrameIdx++;
        m_paintParams.currFrameIdx =
                m_paintParams.currFrameIdx % m_paintParams.nTimeFrames;
    }

//    sdkStartTimer(&m_fpsParams.timer);

    // use OpenGL to build view matrix
    GLfloat modelView[16];
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    float3 viewRot = m_viewportParams.viewRotation;
    glRotatef(-viewRot.x, 1.0, 0.0, 0.0);
    glRotatef(-viewRot.y, 0.0, 1.0, 0.0);
    glRotatef(-viewRot.z, 0.0, 0.0, 1.0);
    float3 viewTrans = m_viewportParams.viewTranslation;
    glTranslatef(-viewTrans.x, -viewTrans.y, -viewTrans.z);
    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
    glPopMatrix();

    m_viewportParams.invViewMatrix[0] = modelView[0];
    m_viewportParams.invViewMatrix[1] = modelView[4];
    m_viewportParams.invViewMatrix[2] = modelView[8];
    m_viewportParams.invViewMatrix[3] = modelView[12];
    m_viewportParams.invViewMatrix[4] = modelView[1];
    m_viewportParams.invViewMatrix[5] = modelView[5];
    m_viewportParams.invViewMatrix[6] = modelView[9];
    m_viewportParams.invViewMatrix[7] = modelView[13];
    m_viewportParams.invViewMatrix[8] = modelView[2];
    m_viewportParams.invViewMatrix[9] = modelView[6];
    m_viewportParams.invViewMatrix[10] = modelView[10];
    m_viewportParams.invViewMatrix[11] = modelView[14];

    render();

    // display results
    glClear(GL_COLOR_BUFFER_BIT);

    // draw image from PBO
    glDisable(GL_DEPTH_TEST);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // draw using texture

    // copy from pbo to texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_texArray[0]);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    m_paintParams.width, m_paintParams.height,
                    GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
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

//    glutSwapBuffers();
//    glutReportErrors();

//    sdkStopTimer(&m_fpsParams.timer);



//    computeFPS();
    //swapBuffers(); http://doc.qt.io/qt-5/qopenglwidget.html#Threading
}

void ExtdOpenGLwidget::initPixelBuffer()
{
    makeCurrent();

    emit sendMsgToGUI(tr("initPixelBuffer."));

    if (m_pbo)
    {
        // unregister this buffer object from CUDA C
        checkCudaErrors(cudaGraphicsUnregisterResource(m_cuda_pbo_resource));

        // delete old buffer
        glDeleteBuffers(1, &m_pbo);
        glDeleteTextures(m_paintParams.nTimeFrames, m_texArray);
    }

    // create pixel buffer object for display
    glGenBuffers(1, &m_pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER,
                 m_paintParams.width*m_paintParams.height*sizeof(GLubyte)*4,
                 0, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_cuda_pbo_resource, m_pbo, cudaGraphicsMapFlagsWriteDiscard));

    m_texArray = new GLuint[m_paintParams.nTimeFrames];

    // create texture for display
    glGenTextures(m_paintParams.nTimeFrames, m_texArray);
    for(int i = 0; i < 1; i++)
    {
        glBindTexture(GL_TEXTURE_2D, m_texArray[i]); // use 0 instead of i ?
        // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_paintParams.width, m_paintParams.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    doneCurrent();
}

void ExtdOpenGLwidget::computeFPS()
{
     emit broadcastFPS(m_fpsParams.fpsCount);
}

void ExtdOpenGLwidget::render()
{

}

void ExtdOpenGLwidget::keyPressEvent(QKeyEvent *_event)
{

}

void ExtdOpenGLwidget::mouseMoveEvent(QMouseEvent *_event)
{

}

void ExtdOpenGLwidget::mouseReleaseEvent(QMouseEvent *_event)
{

}

void ExtdOpenGLwidget::mousePressEvent(QMouseEvent *_event)
{

}

void ExtdOpenGLwidget::wheelEvent(QWheelEvent *_event)
{

}

void ExtdOpenGLwidget::timerEvent(QTimerEvent *)
{

}

int ExtdOpenGLwidget::chooseCudaDevice(bool bUseOpenGL)
{
    int result = -1;

    int c = 1;
    const char* dummy = "device";

    if (bUseOpenGL)
    {
        result = findCudaGLDevice(c, &dummy);
    }
    else
    {
        result = findCudaDevice(c, &dummy);
    }

    return result;
}

void ExtdOpenGLwidget::cleanup()
{
    makeCurrent(); // get context


    sdkDeleteTimer(&m_fpsParams.timer);

    freeCudaBuffers();

    if (m_pbo)
    {
        cudaGraphicsUnregisterResource(m_cuda_pbo_resource);
        glDeleteBuffers(1, &m_pbo);
        glDeleteTextures(m_paintParams.nTimeFrames, m_texArray);
    }

    checkCudaErrors(cudaDeviceSynchronize());

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    printf("Exiting program, resetting device.\n");

    cudaDeviceReset();


    // ... put stuff here


    doneCurrent(); // release context
}

void ExtdOpenGLwidget::loadVolume(QString _loc)
{

}
