#ifndef VOLUMESERVERWIDGET_H
#define VOLUMESERVERWIDGET_H

#include <QWidget>
#include <QString>
#include <QTime>
#include <QTimer>
#include <QDebug>
#include <QSharedPointer>
#include <QThread>

#include <QNetworkInterface>
#include <QTcpServer>
#include <QTcpSocket>
#include <QDataStream>
#include <QByteArray>
#include <QHostAddress>

#include <vector>
#include <memory>

#include "../AscensionWidget/icebot_definitions.h"

namespace Ui {
class VolumeServerWidget;
}

class VolumeServerWidget : public QWidget
{
    Q_OBJECT

public:
    explicit VolumeServerWidget(QWidget *parent = 0);
    ~VolumeServerWidget();
signals:
    void  statusChanged(int event);
    void  tcpError(QAbstractSocket::SocketError error);
    void  newVolumeReceived(QString volume); // change this to Frame type

public slots:
    void  setEpoch(const QDateTime &datetime); // set Epoch
    void  startServer();
    void  stopServer();
    void  newConnectionAvailable();
    void  readVolume();
    void  handleTcpError(QAbstractSocket::SocketError error);
    const QHostAddress getServerAddress() { return m_serverAddress; }
    const quint16 getServerPort() { return m_serverPort; }

private slots:
    void on_toggleServerButton_clicked();

private:
    Ui::VolumeServerWidget *ui;

    // Epoch for time stamps
    // During initializeFrameServer(), check 'isEpochSet' flag
    // If Epoch is set externally from MainWindow, the flag will be true
    // Otherwise, Epoch will be set internally
    QDateTime m_epoch;
    bool m_isEpochSet;

    // Flag to indicate if Frame Server is ready
    // True if initializeFrameServer was successful
    bool m_isReady;

    // Flag to tell that we are still streaming
    bool m_keepStreaming;
    bool m_keepServerRunning;

    // Flag to abort actions (e.g. initialize, acquire, etc.)
    bool m_abort;

    int m_frameCount; // keep a count of number of acquired frames

    // server info
    QHostAddress m_serverAddress;
    quint16 m_serverPort;

    QTcpServer *m_TcpServer;
    QTcpSocket *m_TcpSocket;

    QString m_currVolume;
};

#endif // VOLUMESERVERWIDGET_H
