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
    m_cudaParams.initializedOnce = false;

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
    m_mouseParams.buttonState = Qt::NoButton;

    m_viewportParams.viewTranslation = make_float3(0.0, 0.0, -4.0f);
    m_viewportParams.viewRotation = make_float3(0.0, 0.0, 0.0);

    m_volumeParams.volumeFilename = tr("");
    m_volumeParams.volumeSize = make_cudaExtent(0,0,0);
    m_volumeParams.volumeLoaded = false;

    printf("ExtdOpenGLWidget initialized.\n");

    sdkCreateTimer(&m_fpsParams.timer);

    show();
    update();

}


ExtdOpenGLwidget::~ExtdOpenGLwidget()
{
    //cleanup();
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

    if( (!m_pbo) || (!m_volumeParams.volumeLoaded))
        return;

    sdkStartTimer(&m_fpsParams.timer);

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
//    glTexCoord2f(0, 0);
//    glVertex2f(0, 0);
//    glTexCoord2f(1, 0);
//    glVertex2f(1, 0);
//    glTexCoord2f(1, 1);
//    glVertex2f(1, 1);
//    glTexCoord2f(0, 1);
//    glVertex2f(0, 1);
    glTexCoord2f(0, 0);
    glVertex2f(-1, -1);
    glTexCoord2f(1, 0);
    glVertex2f(1, -1);
    glTexCoord2f(1, 1);
    glVertex2f(1, 1);
    glTexCoord2f(0, 1);
    glVertex2f(-1, 1);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);

//    glutSwapBuffers();
//    glutReportErrors();

    sdkStopTimer(&m_fpsParams.timer);

    computeFPS();
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
    m_fpsParams.frameCount++;
    m_fpsParams.fpsCount++;

    if (m_fpsParams.fpsCount == m_fpsParams.fpsLimit)
    {
        // char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&m_fpsParams.timer) / 1000.f);
        // sprintf(fps, "Volume Render: %3.1f fps, currFrame %d", ifps, currFrame);

        // glutSetWindowTitle(fps);
        m_fpsParams.fpsCount = 0;

        m_fpsParams.fpsLimit = (int)MAX(1.f, ifps);
        sdkResetTimer(&m_fpsParams.timer);
    }

     emit broadcastFPS(m_fpsParams.fpsCount);
}

void ExtdOpenGLwidget::render()
{
    if(!m_volumeParams.volumeLoaded)
        return;

    copyInvViewMatrix(m_viewportParams.invViewMatrix, sizeof(float4)*3);

    // map PBO to get CUDA device pointer
    uint *d_output;
    // map PBO to get CUDA device pointer
    checkCudaErrors(cudaGraphicsMapResources(1, &m_cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output,
                                                         &num_bytes,
                                                         m_cuda_pbo_resource));
    //printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

    // clear image
    checkCudaErrors(cudaMemset(d_output, 0,
                               m_paintParams.width*m_paintParams.height*4));

    // call CUDA kernel, writing results to PBO
    render_kernel(m_cudaParams.gridSize,
                  m_cudaParams.blockSize,
                  d_output,
                  m_paintParams.width,
                  m_paintParams.height,
                  m_raytraceParams.density,
                  m_raytraceParams.brightness,
                  m_raytraceParams.transferOffset,
                  m_raytraceParams.transferScale,
                  m_raytraceParams.lowerThresh,
                  m_paintParams.currFrameIdx);

    getLastCudaError("kernel failed");

    checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cuda_pbo_resource, 0));
}

void ExtdOpenGLwidget::keyPressEvent(QKeyEvent *_event)
{
    _event->accept();

    switch (_event->key())
    {
        case Qt::Key_Escape:
            close();
            return;
            break;

        case Qt::Key_F:
            m_raytraceParams.linearFiltering = !m_raytraceParams.linearFiltering;
            setTextureFilterMode(m_raytraceParams.linearFiltering);
            break;

        case Qt::Key_Plus:
            m_raytraceParams.density += 0.01f;
            break;

        case Qt::Key_Minus:
            m_raytraceParams.density -= 0.01f;
            break;

        case Qt::Key_BracketRight:
            m_raytraceParams.brightness += 0.1f;
            break;

        case Qt::Key_BracketLeft:
            m_raytraceParams.brightness -= 0.1f;
            break;

        case Qt::Key_Apostrophe:
            m_raytraceParams.transferOffset += 0.01f;
            break;

        case Qt::Key_Semicolon:
            m_raytraceParams.transferOffset -= 0.01f;
            break;

        case Qt::Key_Period:
            m_raytraceParams.transferScale += 0.01f;
            break;

        case Qt::Key_Comma:
            m_raytraceParams.transferScale -= 0.01f;
            break;

        case Qt::Key_M:
            m_raytraceParams.lowerThresh += 0.01f;
            break;

        case Qt::Key_N:
            m_raytraceParams.lowerThresh -= 0.01f;
            break;

        case Qt::Key_A:
            m_viewportParams.viewRotation.z += 25 / 5.0f;
            break;

        case Qt::Key_D:
            m_viewportParams.viewRotation.z -= 25 / 5.0f;
            break;

        case Qt::Key_W:
            m_viewportParams.viewRotation.y += 25 / 5.0f;
            break;

        case Qt::Key_S:
            m_viewportParams.viewRotation.y -= 25 / 5.0f;
            break;

        case Qt::Key_Q:
            m_viewportParams.viewRotation.x += 25 / 5.0f;
            break;

        case Qt::Key_E:
            m_viewportParams.viewRotation.x -= 25 / 5.0f;
            break;

        default:
            break;
    }
    update();
}

void ExtdOpenGLwidget::mouseMoveEvent(QMouseEvent *_event)
{
    _event->accept();

    float dx, dy;
    dx = (float)(_event->screenPos().x() - m_mouseParams.ox);
    dy = (float)(_event->screenPos().y() - m_mouseParams.oy);

    if (m_mouseParams.buttonState == Qt::RightButton)
    {
        // right = zoom
        m_viewportParams.viewTranslation.z += dy / 100.0f;
    }
    else if (m_mouseParams.buttonState == Qt::MidButton)
    {
        // middle = translate
        m_viewportParams.viewTranslation.x += dx / 100.0f;
        m_viewportParams.viewTranslation.y -= dy / 100.0f;
    }
    else if (m_mouseParams.buttonState == Qt::LeftButton)
    {
        // left = rotate

        m_viewportParams.viewRotation.x += dy / 5.0f;
        m_viewportParams.viewRotation.y += dx / 5.0f;
    }

    m_mouseParams.ox = _event->screenPos().x();
    m_mouseParams.oy = _event->screenPos().y();

    update();
}

void ExtdOpenGLwidget::mouseReleaseEvent(QMouseEvent *_event)
{
    _event->accept();

    m_mouseParams.buttonState = Qt::NoButton;

    m_mouseParams.ox = _event->screenPos().x();
    m_mouseParams.oy = _event->screenPos().y();

    update();
}

void ExtdOpenGLwidget::mousePressEvent(QMouseEvent *_event)
{
    _event->accept();

    m_mouseParams.buttonState = _event->button();

    m_mouseParams.ox = _event->screenPos().x();
    m_mouseParams.oy = _event->screenPos().y();

    update();
}

void ExtdOpenGLwidget::wheelEvent(QWheelEvent *_event)
{
    _event->accept();

    m_viewportParams.viewTranslation.z += _event->delta() / 120.0f;

    update();
}

void ExtdOpenGLwidget::timerEvent(QTimerEvent *_event)
{
    _event->accept();
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
    std::cout << "Loading volume." << std::endl;
    m_volumeParams.volumeLoaded = false;

    m_volumeParams.volumeFilename = _loc;

    QFile txtfile(_loc + tr("_vol_1.txt"));
    if (txtfile.open(QFile::ReadOnly)) {
        QTextStream txtin(&txtfile);
        txtin >> m_volumeParams.volumeSize.width
                >> m_volumeParams.volumeSize.height
                >> m_volumeParams.volumeSize.depth;
        txtfile.close();
    }
    else
    {
        printf("Error opening txt file: %s!", (_loc + tr("_vol_1.txt")).toStdString().c_str());
        return;
    }

    size_t size = m_volumeParams.volumeSize.width
                 *m_volumeParams.volumeSize.height
                 *m_volumeParams.volumeSize.depth;

    if(h_volume.size() > 0)
    {
        for(void * a : h_volume)
            free(a);
    }


    for(size_t i = 0; i < m_paintParams.nTimeFrames; i++)
    {
        FILE *fp = fopen((_loc+tr("_vol_%1.raw").arg(i+1)).toStdString().c_str(), "rb");

        if (!fp)
        {
            fprintf(stderr, "Error opening file '%s'\n",
                            (_loc+tr("_vol_%1.raw").arg(i+1)).toStdString().c_str());
            return;
        }

        void *dat_ = malloc(size);

        size_t read = fread(dat_, 1, size, fp);
        fclose(fp);

        h_volume.push_back(dat_);

        printf("Read '%s', %d bytes\n",
               (_loc+tr("_vol_%1.raw").arg(i+1)).toStdString().c_str(),
               read);
        std::cout << std::endl;
    }


    if(m_cudaParams.initializedOnce)
    {
        std::cout << "Reinitializing CUDA params." << std::endl;

//        makeCurrent();

//        if (m_pbo)
//        {
//            // unregister this buffer object from CUDA C
//            checkCudaErrors(cudaGraphicsUnregisterResource(m_cuda_pbo_resource));

//            // delete old buffer
//            glDeleteBuffers(1, &m_pbo);
//            glDeleteTextures(m_paintParams.nTimeFrames, m_texArray);
//            m_pbo = 0;
//        }
//        initPixelBuffer();

//        doneCurrent();

        std::cout << "Calling reinitCuda()..." << std::endl;
        reinitCuda(&h_volume[0], m_volumeParams.volumeSize);
        std::cout << "reinitCuda() success." << std::endl;
    }
    else
    {
        initCuda(&h_volume[0], m_volumeParams.volumeSize);
        m_cudaParams.initializedOnce = true;
    }

    m_volumeParams.volumeLoaded = true;
    std::cout << "m_volumeParams.volumeLoaded = true" << std::endl;

    update();
}

void ExtdOpenGLwidget::sliderAction(int actID, double val)
{
    switch(actID)
    {
    case REND_SLIDER_X:
        m_viewportParams.viewRotation.z = val;
        break;

    case REND_SLIDER_Y:
        m_viewportParams.viewRotation.y = val;
        break;

    case REND_SLIDER_Z:
        m_viewportParams.viewRotation.x = val;
        break;

    case REND_SLIDER_DEN:
        m_raytraceParams.density = val;
        break;

    case REND_SLIDER_BRI:
        m_raytraceParams.brightness = val;
        break;

    case REND_SLIDER_TFR_OFF:
        m_raytraceParams.transferOffset = val;
        break;

    case REND_SLIDER_TFR_SCL:
        m_raytraceParams.transferScale = val;
        break;

    case REND_SLIDER_LO_THRESH:
        m_raytraceParams.lowerThresh = val;
        break;

    case REND_SLIDER_LINFILT:
        if(val > 0.5)
            m_raytraceParams.linearFiltering = true;
        else
            m_raytraceParams.linearFiltering = false;
        setTextureFilterMode(m_raytraceParams.linearFiltering);
        break;

    default:
        break;
    }

    update();
}


