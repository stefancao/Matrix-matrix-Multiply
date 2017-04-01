#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <x86intrin.h>

#include "timer.c"

#define N_ 4096
#define K_ 4096
#define M_ 4096
#define BLOCK_SIZE 128

#define MIN(a,b) ( ( a < b) ? a : b )

typedef double dtype;

void matTran(dtype *B, int K, int M){
	dtype BT[K*M];
	for(int i = 0; i < K; i++) {
		for(int j = 0; j < M; j++) {
			BT[j * K + i] = B[i * M + j];
		}
	}
	memcpy(B,BT,K * M * sizeof(dtype));
}

void verify(dtype *C, dtype *C_ans, int N, int M)
{
  int i, cnt;
  cnt = 0;
  for(i = 0; i < N * M; i++) {
    if(abs (C[i] - C_ans[i]) > 1e-6) cnt++;
  }
  if(cnt != 0) printf("ERROR\n"); else printf("SUCCESS\n");
}

// naive
void mm_serial (dtype *C, dtype *A, dtype *B, int N, int K, int M)
{
  int i, j, k;
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < M; j++) {
      for(int k = 0; k < K; k++) {
        C[i * M + j] += A[i * K + k] * B[k * M + j];
      }
    }
  }
}

// cache-blocked matrix-matrix multiply
void mm_cb (dtype *C, dtype *A, dtype *B, int N, int K, int M)
{
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < M; j += BLOCK_SIZE) {
			for(int k = 0; k < K; k += BLOCK_SIZE) {

				// iterate through the blocks
				for (int j_inner = j; j_inner < MIN(j + BLOCK_SIZE, M); j_inner++) {
					for (int k_inner = k; k_inner < MIN(k + BLOCK_SIZE, K); k_inner++) {
						C[i * M + j_inner] += A[i * K + k_inner] * B[k_inner * M + j_inner];
					}
				}
			}
		}
	}
}

// SIMD-vectorized matrix-matrix multiply
void mm_sv (dtype *C, dtype *A, dtype *B, int N, int K, int M)
{
	double c[2];

	// transposing the matrix
	matTran(B,K,M);
	__m128d Avec, Bvec, Cvec, mult_vec; 

	for(int i = 0; i < N; i++) {
		for(int j = 0; j < M; j += BLOCK_SIZE) {
			for(int k = 0; k < K; k += BLOCK_SIZE) {

				// iterate through the blocks
				for (int j_inner = j; j_inner < MIN(j + BLOCK_SIZE, M); j_inner++) {
					
					// init mult_vec
					mult_vec = _mm_setzero_pd();
					for (int k_inner = k; k_inner < MIN(k + BLOCK_SIZE, K); k_inner+=2) {
						
						// load A and B vectors 
						Avec = _mm_load_pd(A + (i * K + k_inner));
						Bvec = _mm_load_pd(B + (j_inner * M + k_inner));

						// compute
						mult_vec = _mm_add_pd(_mm_mul_pd(Avec, Bvec), mult_vec);
					}

					// store
					_mm_store_pd(c, mult_vec);
					C[i * M + j_inner] += c[0] + c[1];
				}
			}
		}
	}
}

int main(int argc, char** argv)
{
  int i, j, k;
  int N, K, M;

  if(argc == 4) {
    N = atoi (argv[1]);		
    K = atoi (argv[2]);		
    M = atoi (argv[3]);		
    printf("N: %d K: %d M: %d\n", N, K, M);
  } else {
    N = N_;
    K = K_;
    M = M_;
    printf("N: %d K: %d M: %d\n", N, K, M);	
  }

  dtype *A = (dtype*) malloc (N * K * sizeof (dtype));
  dtype *B = (dtype*) malloc (K * M * sizeof (dtype));
  dtype *C = (dtype*) malloc (N * M * sizeof (dtype));
  dtype *C_cb = (dtype*) malloc (N * M * sizeof (dtype));
  dtype *C_sv = (dtype*) malloc (N * M * sizeof (dtype));
  assert (A && B && C);

  /* initialize A, B, C */
  srand48 (time (NULL));
  for(i = 0; i < N; i++) {
    for(j = 0; j < K; j++) {
      A[i * K + j] = drand48 ();
    }
  }
  for(i = 0; i < K; i++) {
    for(j = 0; j < M; j++) {
      B[i * M + j] = drand48 ();
    }
  }
  bzero(C, N * M * sizeof (dtype));
  bzero(C_cb, N * M * sizeof (dtype));
  bzero(C_sv, N * M * sizeof (dtype));

  stopwatch_init ();
  struct stopwatch_t* timer = stopwatch_create ();
  assert (timer);
  long double t;

  printf("Naive matrix multiply\n");
  stopwatch_start (timer);
  /* do C += A * B */
  mm_serial (C, A, B, N, K, M);
  t = stopwatch_stop (timer);
  printf("Done\n");
  printf("time for naive implementation: %Lg seconds\n\n", t);


  printf("Cache-blocked matrix multiply\n");
  stopwatch_start (timer);
  /* do C += A * B */
  mm_cb (C_cb, A, B, N, K, M);
  t = stopwatch_stop (timer);
  printf("Done\n");
  printf("time for cache-blocked implementation: %Lg seconds\n", t);

  /* verify answer */
  verify (C_cb, C, N, M);

  printf("SIMD-vectorized Cache-blocked matrix multiply\n");
  stopwatch_start (timer);
  /* do C += A * B */
  mm_sv (C_sv, A, B, N, K, M);
  t = stopwatch_stop (timer);
  printf("Done\n");
  printf("time for SIMD-vectorized cache-blocked implementation: %Lg seconds\n", t);

  /* verify answer */
  verify (C_sv, C, N, M);

  return 0;
}
