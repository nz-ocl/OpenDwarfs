#include "../inc/crc_formats.h"

unsigned char** read_crc(unsigned int* num_pages,unsigned int* page_size,const char* file_path)
{
	FILE* fp;
	unsigned int i,j,read_count=0;
	unsigned char** page;

	fp = fopen(file_path,"r");
	check(fp != NULL,"crc_formats.read_crc() - Cannot Open File");
	fscanf(fp,"%u\n",num_pages);
	fscanf(fp,"%u\n\n",page_size);

	page = malloc(sizeof(void*)*(*num_pages));
	check(page != NULL,"crc_formats.read_crc() - Heap Overflow! Cannot allocate space for page*");

	for(j=0; j<*num_pages; j++)
	{
		page[j] = char_new_array(*page_size,"crc_formats.main() - Heap Overflow! Cannot allocate space for pages");
		for(i=0; i<*page_size; i++)
		  read_count += fscanf(fp,"%hhu ",&page[j][i]);
		check(read_count == *page_size,"crc_formats.read_crc() - Input file corrupted! Read count differs from page size");
		fscanf(fp,"\n");
	}

	fclose(fp);
	return page;
}

void write_crc(const unsigned char** pages, const unsigned int num_pages, const unsigned int page_size,const char* file_path)
{
	FILE* fp;
	int i,j;
	fp = fopen(file_path,"w");
	check(fp != NULL,"crc_formats.write_crc() - Cannot Open File");
	fprintf(fp,"%u\n",num_pages);
	fprintf(fp,"%u\n\n",page_size);

	for(j=0; j<num_pages; j++)
	{
		for(i=0; i<page_size; i++)
		  fprintf(fp,"%hhu ",pages[j][i]);
		fprintf(fp,"\n");
	}

	fclose(fp);
}

unsigned char** rand_crc(const unsigned int num_pages,const unsigned int page_size,const unsigned int seed)
{
	unsigned int i,j;
	unsigned char** page;

	srand(seed);
	page = malloc(sizeof(void*)*num_pages);
	check(page != NULL,"crc_formats.rand_crc() - Heap Overflow! Cannot allocate space for page*");
	for(i=0; i<num_pages; i++)
	{
		page[i] = char_new_array(page_size,"crc_formats.rand_crc() - Heap Overflow! Cannot allocate space for pages");
		for(j=0; j<page_size; j++)
			page[i][j] = rand();
	}
	return page;
}

void free_crc(unsigned char** pages, const unsigned int num_pages)
{
	unsigned int j;
	for(j=0; j<num_pages; j++)
	{
		free(pages[j]);
	}
	free(pages);
}
