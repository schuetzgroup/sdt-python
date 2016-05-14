/*
 *
 * Adaptation of sa_utilities/fitz to accept x and y sigma via stdin
 * for testing purposes and print the result to stdout. This is invoked by
 * z_fit.py
 *
 * 
 * Compilation instructions:
 *
 * Linux:
 *  gcc fitz_test.c -o fitz_test -lm
 *
 */


/* Include */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define MINZ -500
#define MAXZ 500

double *wx_curve;
double *wy_curve;


/*
 * Initialize wx, wy pre-calculated array curves.
 */

void initWxWy(double *wx_params, double *wy_params)
{
  int i,size;
  double zt,sum;

  size = MAXZ-MINZ;

  // wx
  wx_curve = (double *)malloc(sizeof(double)*size);
  for(i=MINZ;i<MAXZ;i++){
    zt = ((double)i - wx_params[1])/wx_params[2];
    sum = 1.0 + zt*zt + wx_params[3]*zt*zt*zt + wx_params[4]*zt*zt*zt*zt;
    sum += wx_params[5]*zt*zt*zt*zt*zt + wx_params[6]*zt*zt*zt*zt*zt*zt;
    wx_curve[i-MINZ] = sqrt(wx_params[0]*sqrt(sum));
  }

  // wy
  wy_curve = (double *)malloc(sizeof(double)*size);
  for(i=MINZ;i<MAXZ;i++){
    zt = ((double)i - wy_params[1])/wy_params[2];
    sum = 1.0 + zt*zt + wy_params[3]*zt*zt*zt + wy_params[4]*zt*zt*zt*zt;
    sum += wy_params[5]*zt*zt*zt*zt*zt + wy_params[6]*zt*zt*zt*zt*zt*zt;
    wy_curve[i-MINZ] = sqrt(wy_params[0]*sqrt(sum));
  }
}


/*
 * Find the best fitting Z value.
 *
 * This just does a grid search.
 */
float findBestZ(double wx, double wy, double cutoff)
{
  int i,best_i;
  double d,dwx,dwy,best_d;

  best_i = 0;
  dwx = wx - wx_curve[0];
  dwy = wy - wy_curve[0];
  best_d = dwx*dwx+dwy*dwy;
  for(i=1;i<(MAXZ-MINZ);i++){
    dwx = wx - wx_curve[i];
    dwy = wy - wy_curve[i];
    d = dwx*dwx+dwy*dwy;
    if(d<best_d){
      best_i = i;
      best_d = d;
    }
  }

  // distance cut-off here
  if (best_d>cutoff){
    return (float)(MINZ-1.0);
  }
  else {
    return (float)(best_i + MINZ);
  }
}


/*
 * Main
 *
 * i3_file - insight3 file on which to perform z calculations.
 * cut_off - distance cutoff
 * wx_params - 7 numbers (wx0, zc, d, A, B, C, D).
 * wy_params - 7 more numbers (wx0, zc, d, A, B, C, D).
 *
 * Expects calibration curves & molecule widths to be in nm.
 */

int main(int argc, const char *argv[])
{
  int i,bad_cat,molecules,offset;
  double cutoff;
  double wx,wy, z;
  double wx_params[7];
  double wy_params[7];

  if (argc != 17){
    fprintf(stderr, "Wrong number of args: %d\n", argc);
    exit(1);
  }

  molecules = strtol(argv[1], NULL, 10);

  // setup
  cutoff = atof(argv[2]);
  cutoff = cutoff*cutoff;

  for(i=3;i<10;i++){
    wx_params[i-3] = atof(argv[i]);
  }

  for(i=10;i<17;i++){
    wy_params[i-10] = atof(argv[i]);
  }

  initWxWy(wx_params, wy_params);

  // analysis
  for(i=0;i<molecules;i++){
      scanf("%lf %lf", &wx, &wy);

      z = findBestZ(sqrt(wx), sqrt(wy), cutoff);
      printf("%lf\n", z);
  }

  // cleanup
  free(wx_curve);
  free(wy_curve);
}


/*
 * The MIT License
 *
 * Copyright (c) 2012 Zhuang Lab, Harvard University
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
