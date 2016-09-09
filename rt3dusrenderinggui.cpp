#include "rt3dusrenderinggui.h"
#include "ui_rt3dusrenderinggui.h"

RT3DUSrenderingGUI::RT3DUSrenderingGUI(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::RT3DUSrenderingGUI)
{
    ui->setupUi(this);

    //m_glwidget = new ExtdOpenGLwidget(this, Qt::Widget);
    m_glwidget = new ExtdOpenGLwidget();

    connect(m_glwidget, SIGNAL(sendMsgToGUI(QString)),
            this, SLOT(receiveMsg(QString)));

    //ui->SEASlogo->setPixmap(QPixmap("C:\\Users\\Alperen\\Documents\\QT Projects\\RT3DUSrenderingGUI\\SEASLogo.png"));
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
