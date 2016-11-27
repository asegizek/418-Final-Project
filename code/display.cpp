#include <stdio.h>
#include <stdlib.h>

#include "platformgl.h"

#include "automata.h"


GLint width = 640;
GLint height = 480;
//GLint rows = 20;
//GLint cols = 20;
GLfloat left = 0.0;
GLfloat right = 1.0;
GLfloat bottom = 1.0;
GLfloat top = 0.0;

static struct {
    Automaton* automaton;
    int rows;
    int cols;
} gData;

void renderFrame(){
    return;

}

void reshape(int w, int h) {
	width = w;
	height = h;

	glViewport(0, 0, width,height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(left, right, bottom, top);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glutPostRedisplay();
}


void displaySquares(void) {
    
    glClear(GL_COLOR_BUFFER_BIT);
    glColor3f(0.0, 1.0, 0.0);
    int rows = gData.rows;
    int cols = gData.cols;
    float xSize = 1.0f/ ((float)cols);
    float ySize = 1.0f / ((float)rows);
    float size = 0.05f;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            float g = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            float b = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            glColor3f(r, g, b);
            float cx = (j+0.5f)*xSize ;
            float cy = (i+0.5f)*ySize;
            
            glBegin(GL_POLYGON);
            glVertex2f(cx-xSize/2, cy-ySize/2);
            glVertex2f(cx+xSize/2, cy-ySize/2);
            glVertex2f(cx+xSize/2, cy+ySize/2);
            glVertex2f(cx-xSize/2, cy+ySize/2);
            glEnd();
            
        }
    }
    glFlush();
    glutSwapBuffers();
    glutPostRedisplay();
}

void displayHexagons(void) {
    glClear(GL_COLOR_BUFFER_BIT);
    glColor3f(0.0, 1.0, 0.0);
    int rows = gData.rows;
    int cols = gData.cols;
    float h_width = 1.0f/ ((float)(cols+0.5f));
    //quarter of the hexagon height
    float h_qheight = 1.0f / (float)(4+(rows-1)*3);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float cx = (i%2 == 0) ? (j+0.5f)*h_width : (j+1.0f)*h_width;
            float cy = (2+3*i)*h_qheight;
            glColor3f((float)i/rows, (float)j/cols, (float)(i+j)/(cols+rows));
            glBegin(GL_POLYGON);
            glVertex2f(cx-h_width/2, cy-h_qheight);
            glVertex2f(cx          , cy-2*h_qheight);
            glVertex2f(cx+h_width/2, cy-h_qheight);
            glVertex2f(cx+h_width/2, cy+h_qheight);
            glVertex2f(cx          , cy+2*h_qheight);
            glVertex2f(cx-h_width/2, cy+h_qheight);
            glEnd();
        }
    }
    glFlush();
    glutSwapBuffers();
    glutPostRedisplay();
}
void displayHexagonsSlanted(void) {
    glClear(GL_COLOR_BUFFER_BIT);
    glColor3f(0.0, 1.0, 0.0);
    int rows = gData.rows;
    int cols = gData.cols;
    float h_width = 1.0f/ ((float)(cols + (rows - 1) * 0.5f));
    //quarter of the hexagon height
    float h_qheight = 1.0f / (float)(4+(rows-1)*3);
    Grid* grid = gData.automaton->get_grid();
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float cx = (j + (i + 1) * 0.5f) * h_width;
            float cy = (2+3*i)*h_qheight;
            int grid_val = grid->data[i*rows + j];
            float r = (grid_val) ? 0.0f : 1.0f;
            glColor3f(r, 1.0f, 1.0f);
            glBegin(GL_POLYGON);
            glVertex2f(cx-h_width/2, cy-h_qheight);
            glVertex2f(cx          , cy-2*h_qheight);
            glVertex2f(cx+h_width/2, cy-h_qheight);
            glVertex2f(cx+h_width/2, cy+h_qheight);
            glVertex2f(cx          , cy+2*h_qheight);
            glVertex2f(cx-h_width/2, cy+h_qheight);
            glEnd();
            glColor3f(0.0f, 0.0f, 0.0f);
            glBegin(GL_LINE_LOOP);
            glVertex2f(cx-h_width/2, cy-h_qheight);
            glVertex2f(cx          , cy-2*h_qheight);
            glVertex2f(cx+h_width/2, cy-h_qheight);
            glVertex2f(cx+h_width/2, cy+h_qheight);
            glVertex2f(cx          , cy+2*h_qheight);
            glVertex2f(cx-h_width/2, cy+h_qheight);
            glEnd();

        }
    }
    glFlush();
    glutSwapBuffers();
    glutPostRedisplay();
}

void handleKeyPress(unsigned char key, int x , int y) {
    switch(key) {
    case 'n':
        gData.automaton->update_cells();
        break;
    }
}


void startRenderer(Automaton* automaton, int rows, int cols) {
    gData.automaton = automaton;
    gData.rows = rows;
    gData.cols = cols;
    glutCreateWindow("Drawing hexagons");
    glutInitWindowSize(1280, 960);

    glutDisplayFunc(displayHexagonsSlanted);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(handleKeyPress);
    glutMainLoop();
}

/*
int main(int argc, char **argv) {
    
    width  = 640;
    height = 480;
    
    glutInit(&argc, argv);
    glutInitWindowSize(width, height);
    glutCreateWindow("Drawing hexagons");
    
    glutDisplayFunc(displayHexagons);
    glutReshapeFunc(reshape);
    glutMainLoop();
    return 0;
}
*/