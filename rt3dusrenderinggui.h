#ifndef RT3DUSRENDERINGGUI_H
#define RT3DUSRENDERINGGUI_H

#include <QMainWindow>
#include "extdopenglwidget.h"

namespace Ui {
class RT3DUSrenderingGUI;
}

class RT3DUSrenderingGUI : public QMainWindow
{
    Q_OBJECT

public:
    explicit RT3DUSrenderingGUI(QWidget *parent = 0);
    ~RT3DUSrenderingGUI();

public slots:
    void receiveMsg(QString msg);

private:
    Ui::RT3DUSrenderingGUI *ui;

    ExtdOpenGLwidget *m_glwidget;

};

#endif // RT3DUSRENDERINGGUI_H
