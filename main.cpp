#include "rt3dusrenderinggui.h"
#include <QApplication>

int main(int argc, char *argv[])
{

    QApplication a(argc, argv);

    RT3DUSrenderingGUI w;

    w.show();

    return a.exec();
}
