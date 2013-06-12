/*
 */

void __kernel csr(const unsigned int num_rows,
                       __global unsigned int * Ap, 
                       __global unsigned int * Aj, 
                       __global float * Ax, 
                       __global float * x, 
                       __global float * y)
{
  	unsigned int row = get_global_id(0);
	unsigned int next_nz_row;
	
    if(row < num_rows)
    {     
        float sum = y[row];
        const unsigned int row_start = Ap[row];
        if(row_start == -1) return; //If there aren't any non-zero elements in this row, y[row] already contains the proper value
        	
      	next_nz_row = row + 1; //Initialize search for next non-zero row with row following the row of interest
      	while(Ap[next_nz_row] == -1 && next_nz_row < num_rows) next_nz_row++; //Check every remaining row for value other than -1 (non-zero sentinel)
      	
       	const unsigned int row_end = Ap[next_nz_row]; //row_end now contains index into Ax of first element not within row of interest

        
        unsigned int jj = 0;
        for (jj = row_start; jj < row_end; jj++)
            sum += Ax[jj] * x[Aj[jj]];      

        y[row] = sum;
    }
	
}
