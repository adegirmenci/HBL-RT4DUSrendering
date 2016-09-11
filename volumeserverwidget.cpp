#include "volumeserverwidget.h"
#include "ui_volumeserverwidget.h"

VolumeServerWidget::VolumeServerWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::VolumeServerWidget)
{
    ui->setupUi(this);

    m_isEpochSet = false;
    m_keepStreaming = false;
    m_keepServerRunning = false;
    m_frameCount = 0;
    m_abort = false;
    m_serverAddress = QHostAddress(QHostAddress::LocalHost);
    m_serverPort = (quint16)4419;

    m_TcpSocket = Q_NULLPTR;

    m_TcpServer = new QTcpServer(this);
    //m_TcpServer->setMaxPendingConnections(1);

    connect(m_TcpServer, SIGNAL(newConnection()),
            this, SLOT(newConnectionAvailable()));
    connect(m_TcpServer, SIGNAL(acceptError(QAbstractSocket::SocketError)),
            this, SLOT(handleTcpError(QAbstractSocket::SocketError)));
    connect(this, SIGNAL(tcpError(QAbstractSocket::SocketError)),
            this, SLOT(handleTcpError(QAbstractSocket::SocketError)));

    m_isReady = true;
}

VolumeServerWidget::~VolumeServerWidget()
{
    m_abort = true;
    m_isReady = false;
    m_keepStreaming = false;
    stopServer();

    qDebug() << "Closing VolumeServerWidget - Thread ID: " << QThread::currentThreadId() << ".";

    delete ui;
}

void VolumeServerWidget::startServer()
{
    if(m_isReady)
    {
        if(!m_TcpServer->listen(m_serverAddress, m_serverPort))
        {
            emit tcpError(m_TcpServer->serverError());
    //        qDebug() << tr("FrameServerThread: Unable to start the server: %1.")
    //                    .arg(m_TcpServer->errorString());
            emit statusChanged(VOLSRVR_START_FAILED);
            ui->toggleServerButton->setText("Start Server");
            m_keepServerRunning = false;
        }
        else
        {
            emit statusChanged(VOLSRVR_STARTED);
            ui->toggleServerButton->setText("Stop Server");
            ui->addrPortLineEdit->setText(tr("%1:%2")
                                          .arg(getServerAddress().toString())
                                          .arg(getServerPort()));
            m_keepServerRunning = true;
        }
    }
    else
    {
        emit statusChanged(VOLSRVR_START_FAILED);
        ui->toggleServerButton->setText("Start Server");
        m_keepServerRunning = false;
    }

}

void VolumeServerWidget::stopServer()
{
    //m_TcpSocket->flush();
    //m_TcpSocket->close();
    if(m_TcpServer)
    {
        m_TcpServer->close();

        if(m_TcpServer->isListening())
        {
            emit statusChanged(VOLSRVR_CLOSE_FAILED);
            ui->toggleServerButton->setText("Stop Server");
            m_keepServerRunning = true;
        }
        else
        {
            emit statusChanged(VOLSRVR_CLOSED);
            ui->toggleServerButton->setText("Start Server");
            m_keepServerRunning = false;
        }
    }
    else
    {
        emit statusChanged(VOLSRVR_CLOSED);
        ui->toggleServerButton->setText("Start Server");
        m_keepServerRunning = false;
    }
}

void VolumeServerWidget::newConnectionAvailable()
{
    emit statusChanged(VOLSRVR_NEW_CONNECTION);

    m_TcpSocket = m_TcpServer->nextPendingConnection();
    connect(m_TcpSocket, SIGNAL(readyRead()), this, SLOT(readVolume()));
    connect(m_TcpSocket, SIGNAL(error(QAbstractSocket::SocketError)),
            this, SLOT(handleTcpError(QAbstractSocket::SocketError)));
    connect(m_TcpSocket, SIGNAL(disconnected()),
            m_TcpSocket, SLOT(deleteLater()));

    if(m_TcpSocket->isReadable())
    {
        qDebug() << "VolumeServerWidget: Socket is readable.";
    }
    else {
        emit statusChanged(VOLSRVR_SOCKET_NOT_READABLE); }

}

void VolumeServerWidget::readVolume()
{
    QDataStream in(m_TcpSocket);
    in.setVersion(QDataStream::Qt_5_7);

    quint16 blockSize = 0;

    if (blockSize == 0) {
        if (m_TcpSocket->bytesAvailable() < (int)sizeof(quint16))
            return;
        in >> blockSize;
    }

    if (in.atEnd())
        return;

    QString nextVolume;
    in >> nextVolume;

    if (nextVolume == m_currVolume) {
        return;
    }

    m_currVolume = nextVolume;
    emit newVolumeReceived(m_currVolume);
    qDebug() << "Received: " << m_currVolume;
    emit VOLSRVR_VOLUME_RECEIVED;
}

void VolumeServerWidget::handleTcpError(QAbstractSocket::SocketError error)
{
    QString errStr;
    switch(error)
    {
    case QAbstractSocket::ConnectionRefusedError:
        errStr = "ConnectionRefusedError"; break;
    case QAbstractSocket::RemoteHostClosedError:
        errStr = "RemoteHostClosedError"; break;
    case QAbstractSocket::HostNotFoundError:
        errStr = "HostNotFoundError"; break;
    case QAbstractSocket::SocketAccessError:
        errStr = "SocketAccessError"; break;
    case QAbstractSocket::SocketResourceError:
        errStr = "SocketResourceError"; break;
    case QAbstractSocket::SocketTimeoutError:
        errStr = "SocketTimeoutError"; break;
    case QAbstractSocket::DatagramTooLargeError:
        errStr = "DatagramTooLargeError"; break;
    case QAbstractSocket::NetworkError:
        errStr = "NetworkError"; break;
    case QAbstractSocket::AddressInUseError:
        errStr = "AddressInUseError"; break;
    case QAbstractSocket::SocketAddressNotAvailableError:
        errStr = "SocketAddressNotAvailableError"; break;
    case QAbstractSocket::UnsupportedSocketOperationError:
        errStr = "UnsupportedSocketOperationError"; break;
    case QAbstractSocket::OperationError:
        errStr = "OperationError"; break;
    case QAbstractSocket::TemporaryError:
        errStr = "TemporaryError"; break;
    case QAbstractSocket::UnknownSocketError:
        errStr = "UnknownSocketError"; break;
    default:
        errStr = "UnknownError";
    }

    qDebug() << tr("Error in VolumeServerWidget: %1.")
                   .arg(errStr);
}

void VolumeServerWidget::setEpoch(const QDateTime &datetime)
{
    if(!m_keepStreaming)
    {
        m_epoch = datetime;
        m_isEpochSet = true;

//        emit logEventWithMessage(SRC_VOLSRVR, LOG_INFO, QTime::currentTime(), VOLSRVR_EPOCH_SET,
//                                 m_epoch.toString("yyyy/MM/dd - hh:mm:ss.zzz"));
    }
//    else
//        emit logEvent(SRC_VOLSRVR, LOG_INFO, QTime::currentTime(), VOLSRVR_EPOCH_SET_FAILED);
}

void VolumeServerWidget::on_toggleServerButton_clicked()
{
    if(m_keepServerRunning) {
        stopServer(); }
    else {
        startServer(); }
}
