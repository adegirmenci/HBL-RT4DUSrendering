#ifndef RT3DUSRENDERINGGUI_H
#define RT3DUSRENDERINGGUI_H

#include <QMainWindow>
#include <QFileDialog>
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

signals:
    void sliderAction(int actID, double val);
    void loadVolume(QString fileName);

public slots:
    void receiveMsg(QString msg);

    void loadVolumeClicked();

private slots:
    void serverStatusChanged(int status);

    void on_xRotSlider_valueChanged(int value);

    void on_yRotSlider_valueChanged(int value);

    void on_zRotSlider_valueChanged(int value);

    void on_lowerThreshSlider_valueChanged(int value);

    void on_upperThreshSlider_valueChanged(int value);

    void on_transferoffsetSlider_valueChanged(int value);

    void on_transferScaleSlider_valueChanged(int value);

    void on_densitySlider_valueChanged(int value);

    void on_brightnessSlider_valueChanged(int value);

    void on_linFiltCheckBox_toggled(bool checked);

private:
    Ui::RT3DUSrenderingGUI *ui;

    ExtdOpenGLwidget *m_glwidget;

};

#endif // RT3DUSRENDERINGGUI_H
