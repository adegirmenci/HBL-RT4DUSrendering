#include "rt3dusrenderinggui.h"
#include "ui_rt3dusrenderinggui.h"

RT3DUSrenderingGUI::RT3DUSrenderingGUI(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::RT3DUSrenderingGUI)
{
    ui->setupUi(this);

    //m_glwidget = new ExtdOpenGLwidget();
    m_glwidget = new ExtdOpenGLwidget(QOpenGLWindow::NoPartialUpdate, this->windowHandle());

    m_glwidget->setTitle(tr("Volume Rendering"));

    connect(m_glwidget, SIGNAL(sendMsgToGUI(QString)),
            this, SLOT(receiveMsg(QString)));

    // close this if GL window is closed - might not be good in the long run
    connect(m_glwidget, SIGNAL(visibleChanged(bool)),
            this, SLOT(close()));

    ui->SEASlogo->setPixmap(QPixmap("C:\\Users\\Alperen\\Documents\\QT Projects\\RT3DUSrenderingGUI\\SEASLogo.png"));

    connect(ui->volumeServerWidget, SIGNAL(statusChanged(int)),
            this, SLOT(serverStatusChanged(int)));
    connect(ui->volumeServerWidget, SIGNAL(newVolumeReceived(QString)),
            m_glwidget, SLOT(loadVolume(QString)));
}

RT3DUSrenderingGUI::~RT3DUSrenderingGUI()
{
    m_glwidget->close();
    //m_glwidget->destroy();
    delete m_glwidget;//->deleteLater();
    std::cout << "BYE" << std::endl;
    delete ui;
}

void RT3DUSrenderingGUI::receiveMsg(QString msg)
{
    ui->statusTextEdit->appendPlainText(msg);
}

void RT3DUSrenderingGUI::serverStatusChanged(int status)
{
    switch(status)
    {
    case VOLSRVR_STARTED:
        ui->statusTextEdit->appendPlainText("Server started.");
        break;
    case VOLSRVR_START_FAILED:
        ui->statusTextEdit->appendPlainText("Server start failed.");
        break;
    case VOLSRVR_CLOSED:
        ui->statusTextEdit->appendPlainText("Server closed.");
        break;
    case VOLSRVR_CLOSE_FAILED:
        ui->statusTextEdit->appendPlainText("Server stop failed.");
        break;
    case VOLSRVR_NEW_CONNECTION:
        ui->statusTextEdit->appendPlainText("Incoming connection.");
        break;
    case VOLSRVR_SOCKET_NOT_READABLE:
        ui->statusTextEdit->appendPlainText("Socket not readable.");
        break;
    case VOLSRVR_VOLUME_RECEIVED:
        ui->statusTextEdit->appendPlainText("Received volume.");
        break;
    default:
        ui->statusTextEdit->appendPlainText("Unknown server state.");
    }
}
