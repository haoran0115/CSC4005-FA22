#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <GL/freeglut.h>


void guiExit(unsigned char key, int x, int y){
    switch ( key )
    {
    case 27:
        exit(0);
        break; 
    }
    glutPostRedisplay();
}

