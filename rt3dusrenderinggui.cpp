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

    connect(this, SIGNAL(sliderAction(int,double)),
            m_glwidget, SLOT(sliderAction(int,double)));
    connect(this, SIGNAL(loadVolume(QString)),
            m_glwidget, SLOT(loadVolume(QString)));

    connect(ui->actionLoad_Volume, SIGNAL(triggered(bool)),
            this, SLOT(loadVolumeClicked()));
}

RT3DUSrenderingGUI::~RT3DUSrenderingGUI()
{
    m_glwidget->deleteLater();
    std::cout << "BYE" << std::endl;
    delete ui;
}

void RT3DUSrenderingGUI::closeEvent(QCloseEvent *event)
{
    event->accept();

    m_glwidget->close();
    m_glwidget->destroy();
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

void RT3DUSrenderingGUI::on_xRotSlider_valueChanged(int value)
{
    emit sliderAction(REND_SLIDER_X, value);
}

void RT3DUSrenderingGUI::on_yRotSlider_valueChanged(int value)
{
    emit sliderAction(REND_SLIDER_Y, value);
}

void RT3DUSrenderingGUI::on_zRotSlider_valueChanged(int value)
{
    emit sliderAction(REND_SLIDER_Z, value);
}

void RT3DUSrenderingGUI::on_lowerThreshSlider_valueChanged(int value)
{
    if(value > ui->upperThreshSlider->value())
    {
        ui->upperThreshSlider->setValue(value);
        emit sliderAction(REND_SLIDER_HI_THRESH, value*0.1);
    }
    emit sliderAction(REND_SLIDER_LO_THRESH, value*0.1);
}

void RT3DUSrenderingGUI::on_upperThreshSlider_valueChanged(int value)
{
    if(value < ui->lowerThreshSlider->value())
    {
        ui->lowerThreshSlider->setValue(value);
        emit sliderAction(REND_SLIDER_LO_THRESH, value*0.1);
    }
    emit sliderAction(REND_SLIDER_HI_THRESH, value*0.1);
}

void RT3DUSrenderingGUI::on_transferoffsetSlider_valueChanged(int value)
{
    emit sliderAction(REND_SLIDER_TFR_OFF, value*0.05);
}

void RT3DUSrenderingGUI::on_transferScaleSlider_valueChanged(int value)
{
    emit sliderAction(REND_SLIDER_TFR_SCL, value*0.05);
}

void RT3DUSrenderingGUI::on_densitySlider_valueChanged(int value)
{
    emit sliderAction(REND_SLIDER_DEN, value*0.05);
}

void RT3DUSrenderingGUI::on_brightnessSlider_valueChanged(int value)
{
    emit sliderAction(REND_SLIDER_BRI, value*0.05);
}

void RT3DUSrenderingGUI::on_linFiltCheckBox_toggled(bool checked)
{
    emit sliderAction(REND_SLIDER_LINFILT, checked*1.0);
}

void RT3DUSrenderingGUI::loadVolumeClicked()
{
    QString file = QFileDialog::getOpenFileName(this, tr("Choose Volume File"),
                                               "../",
                                               tr("3D Volume Definition (*.txt)"));;

    ui->statusTextEdit->appendPlainText(file);

    emit loadVolume(file.left(file.length()-10));
}
