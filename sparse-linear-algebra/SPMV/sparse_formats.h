/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*
 *  Compressed Sparse Row matrix (aka CRS)
 * valueType = float, IndexType = unsigned int
 */

#define MAX(a,b) (a > b ? a:b)
#define CSR_EMPTY_ROW -1

typedef struct csr_matrix
{
    unsigned int index_type;
    float value_type;
    unsigned int num_rows, num_cols, num_nonzeros;

    unsigned int * Ap;  //row pointer
    unsigned int * Aj;  //column indices
    float * Ax;  //nonzeros
}
csr_matrix;

typedef struct triplet
{
	unsigned int i_type,j_type;
	float value_type;

	unsigned int i,j;
	float v;
}
triplet;

typedef struct coo_matrix
{
	unsigned int index_type;
	float value_type;
	unsigned long density_ppm;
	unsigned int num_rows,num_cols,num_nonzeros;

	triplet* non_zero;
}
coo_matrix;

unsigned int * int_new_array(const size_t N) 
{ 
    //dispatch on location
    return (unsigned int*) malloc(N * sizeof(unsigned int));
}

float * float_new_array(const size_t N) 
{ 
    //dispatch on location
    return (float*) malloc(N * sizeof(float));
}

triplet* triplet_new_array(const size_t N)
{
	//dispatch on location
	return (triplet*) malloc(N * sizeof(triplet));
}

/*
 * Comparator function implemented for use with qsort
 *
 * Could a speedup be achieved by using qsort as the
 * intermediate sort for radix sort if two separate
 * comparators (one for each coordinate [i,j]) were
 * implemented?
 */
int triplet_comparator(const void *v1, const void *v2)
{
	const triplet* t1 = (triplet*) v1;
	const triplet* t2 = (triplet*) v2;

	if(t1->i < t2->i)
		return -1;
	else if(t1->i > t2->i)
		return +1;
	else if(t1->j < t2->j)
		return -1;
	else if(t1->j > t2->j)
		return +1;
	else
		return 0;
}

/*
 * Generate random integer between high_bound & low_bound, inclusive.
 *
 * preconditions: 	HB > 0
 * 					LB > 0
 * 					HB >= LB
 * Returns: -1 if preconditions not met, random int within [LB,HB] otherwise.
 */
int gen_rand(const int LB, const int HB)
{
	int range = HB - LB + 1;
	if(HB < 0 || LB < 0 || range <= 0)
			return -1;
    return (rand() % range) + LB;
}

/*
 * The standard 5-point finite difference approximation
 * to the Laplacian operator on a regular N-by-N grid.
 */

csr_matrix laplacian_5pt(const unsigned int N)
{
    csr_matrix csr;
    csr.num_rows = N*N;
    csr.num_cols = N*N;
    csr.num_nonzeros = 5*N*N - 4*N; 

    csr.Ap = int_new_array(csr.num_rows+4);

    csr.Aj = int_new_array(csr.num_nonzeros);
    
    csr.Ax = float_new_array(csr.num_nonzeros);

    unsigned int nz = 0;
    unsigned int i = 0;
    unsigned int j = 0;
    unsigned int indx = 0;

    for(i = 0; i < N; i++)
    {
        for(j = 0; j < N; j++)
        {
            indx = N*i + j;

            if (i > 0){
                csr.Aj[nz] = indx - N;
                csr.Ax[nz] = -1;
                nz++;
            }

            if (j > 0){
                csr.Aj[nz] = indx - 1;
                csr.Ax[nz] = -1;
                nz++;
            }

            csr.Aj[nz] = indx;
            csr.Ax[nz] = 4;
            nz++;

            if (j < N - 1){
                csr.Aj[nz] = indx + 1;
                csr.Ax[nz] = -1;
                nz++;
            }

            if (i < N - 1){
                csr.Aj[nz] = indx + N;
                csr.Ax[nz] = -1;
                nz++;
            }
            
            csr.Ap[indx + 1] = nz;
        }
    }
    return csr;
}


/*
 * Method to generate a random matrix in COO form of given size and density
 *
 * N = L&W of square matrix
 * density = density (fraction of NZ elements) expressed in parts per million (ppm)
 *
 * returns coo_matrix struct
 */
coo_matrix rand_square_coo(const unsigned int N,const unsigned long density)
{
	coo_matrix coo;
	triplet* current_triplet;
	triplet* preexisting_triplet;
	unsigned int ind,ind2;

	coo.num_rows = N;
	coo.num_cols = N;
	coo.density_ppm = density;
	coo.num_nonzeros = N*N*density/1000000;

	coo.non_zero = triplet_new_array(coo.num_nonzeros);

	for(ind=0; ind<coo.num_nonzeros; ind++)
	{
		current_triplet = &(coo.non_zero[ind]);
		(current_triplet->i) = gen_rand(0,N-1);
		(current_triplet->j) = gen_rand(0,N-1);
		for(ind2=0; ind2<ind; ind2++)
		{
			preexisting_triplet = &(coo.non_zero[ind2]);
			if((current_triplet->i) == (preexisting_triplet->i) && (current_triplet->j) == (preexisting_triplet->j))
			{
				ind--;
				break;
			}
		}
	}

	for(ind=0; ind<coo.num_nonzeros; ind++)
	{
		current_triplet = &(coo.non_zero[ind]);
		while((current_triplet->v) == 0.0)
				(current_triplet->v) = 1.0 - 2.0 * (rand() / (2147483647 + 1.0));
	}

	qsort(coo.non_zero,coo.num_nonzeros,sizeof(triplet),triplet_comparator);

	return coo;
}

csr_matrix coo_to_csr(const coo_matrix* coo)
{
	int ind,row_count,newline_count;

	csr_matrix csr;
	csr.num_rows = coo->num_rows;
	csr.num_cols = coo->num_cols;
	csr.num_nonzeros = coo->num_nonzeros;

	csr.Ap = int_new_array(csr.num_rows+1);
	csr.Aj = int_new_array(csr.num_nonzeros);
	csr.Ax = float_new_array(csr.num_nonzeros);

	for(ind=0; ind<coo->num_nonzeros; ind++)
	{
		csr.Ax[ind] = coo->non_zero[ind].v;
		csr.Aj[ind] = coo->non_zero[ind].j;
	}

	row_count = 0;
	ind=0;
	while(ind < coo->non_zero[0].i)
	{
		csr.Ap[row_count++] = -1;
		ind++;
	}
	csr.Ap[row_count++] = 0;

	for(ind=1; ind<coo->num_nonzeros; ind++)
	{
		newline_count = coo->non_zero[ind].i - coo->non_zero[ind-1].i;
		while(newline_count > 1)
		{
			csr.Ap[row_count++] = CSR_EMPTY_ROW;
			newline_count--;
		}
		if(newline_count == 1)
			csr.Ap[row_count++] = ind;
	}

	csr.Ap[row_count] = csr.num_nonzeros;
	return csr;
}




