#ifndef EXTDOPENGLWIDGET_H
#define EXTDOPENGLWIDGET_H

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <QObject>
#include <QOpenGLWidget>
#include <QTime>
#include <QOpenGLBuffer>
#include <QOpenGLTexture>
#include <QOpenGLContext>
#include <QDebug>
#include <QOpenGLWindow>
#include <QOpenGLFunctions>
#include <QWindow>

#include <QKeyEvent>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QTimerEvent>

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

#include "rt3dus_definitions.h"

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
void reinitCuda(void **h_volume, cudaExtent volumeSize);

extern "C"
void freeCudaBuffers();

extern "C"
void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH,
                              float density, float brightness, float transferOffset, float transferScale, float lowerThresh, int currFrame);
extern "C"
void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix);

struct RaytraceParams {
    float density;
    float brightness;
    float transferOffset;
    float transferScale;
    float lowerThresh;
    bool linearFiltering;
};

struct PaintParams {
    unsigned int nTimeFrames;
    unsigned int currFrameIdx;
    unsigned int frameGaps;
    unsigned int framesShown;
    uint width, height;
};

struct CudaParams {
    dim3 blockSize;
    dim3 gridSize;
    bool initializedOnce;
};

struct VolumeParams {
    cudaExtent volumeSize;
    QString volumeFilename;
    bool volumeLoaded;
};

struct ViewportParams {
    float3 viewRotation;
    float3 viewTranslation;
    float invViewMatrix[12];
};

struct FPSparams {
    StopWatchInterface *timer;
    int frameCheckNumber;
    int fpsCount;        // FPS count for averaging
    int fpsLimit;        // FPS limit for sampling
    int g_Index;
    unsigned int frameCount;
};

struct MouseParams {
    int ox, oy;
    Qt::MouseButton buttonState;
};


class ExtdOpenGLwidget : public QOpenGLWindow, protected QOpenGLFunctions
{
    Q_OBJECT
public:
    //explicit ExtdOpenGLwidget(QWidget* parent = Q_NULLPTR, Qt::WindowFlags f = Qt::WindowFlags());
    explicit ExtdOpenGLwidget(UpdateBehavior updateBehavior = NoPartialUpdate, QWindow *parent = Q_NULLPTR);
    ~ExtdOpenGLwidget();

    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();

    void initPixelBuffer();
    void computeFPS();
    void render();

    void keyPressEvent(QKeyEvent *_event);
    void mouseMoveEvent(QMouseEvent *_event);
    void mouseReleaseEvent(QMouseEvent *_event);
    void mousePressEvent(QMouseEvent *_event);
    void wheelEvent(QWheelEvent *_event);
    void timerEvent(QTimerEvent *_event);

    int iDivUp(int a, int b) {return (a % b != 0) ? (a / b + 1) : (a / b); }
    int chooseCudaDevice(bool bUseOpenGL);

signals:
    void sendMsgToGUI(QString msg);
    void broadcastFPS(int FPS);
    void updateGUIelement(int element, double value);

public slots:
    void cleanup();

    void loadVolume(QString _loc);
    void setDensity(float _den) { m_raytraceParams.density = _den; }

    void sliderAction(int actID, double val);

private:
    RaytraceParams m_raytraceParams;
    PaintParams m_paintParams;
    CudaParams m_cudaParams;
    ViewportParams m_viewportParams;
    FPSparams m_fpsParams;
    MouseParams m_mouseParams;
    VolumeParams m_volumeParams;

    QTime m_currentTime;

    //QOpenGLWindow *win;

    //std::vector<uchar> h_volume;
    std::vector<void *> h_volume;

    //QOpenGLBuffer m_pbo;
    //QOpenGLTexture m_texArray;
    GLuint m_pbo; // OpenGL pixel buffer object
    GLuint *m_texArray;    // OpenGL texture object
    struct cudaGraphicsResource *m_cuda_pbo_resource;  // CUDA Graphics Resource (to transfer PBO)

};

#endif // EXTDOPENGLWIDGET_H
