#include <stdio.h>
int peek(void) { FILE *f = fopen("anything", "r"); return f != 0; }
