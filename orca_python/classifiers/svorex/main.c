/*******************************************************************************\

	main.c in Sequential Minimal Optimization ver2.0
		
	entry function.
		
	Chu Wei Copyright(C) National Univeristy of Singapore
	Create on Jan. 16 2000 at Control Lab of Mechanical Engineering 
	Update on Aug. 23 2001 

\*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#ifndef __MACH__
    #include <malloc.h>
#endif
#include <string.h>
#include <limits.h>
#include <math.h>
#include <time.h>
#include "smo.h"
#define VERSION (0)

int main( int argc, char * argv[])
{
	def_Settings * defsetting = NULL ;
	smo_Settings * smosetting = NULL ;
	//kcv_Settings * kcvsetting ;
	//Data_Node * node ;
	char buf[LENGTH] ;
	char filename[1024] ;
	unsigned int sz = 0;
	unsigned int index = 0 ;
	double parameter = 0 ;
	FILE * log ; 
	//double * guess ;

	printf("\nSupport Vector Ordinal Regression Using K-fold Cross Validation v2.%d \n--- Chu Wei Copyright(C) 2003-2004\n\n", VERSION) ;
	if ( 1 == argc || NULL == (defsetting = Create_def_Settings(argv[--argc])) )
	{
		// display help	
		printf("\nUsage:  svms [-v] [...] [-K k] file \n\n") ;
		printf("  file   specifies the file containing training samples.\n") ;
		printf("  -v     activates the verbose mode to display message.\n") ;	
		printf("  -L     use imbalanced Linear kernel (default Gaussian kernel).\n") ;
		printf("  -P  p  use Polynomial kernel with order p (default Gaussian kernel).\n") ;
		printf("  -E  e  set Epsilon at e for regression only (default 0.1).\n") ;					
		printf("  -i     normalize the training inputs.\n") ;		
		printf("  -o     normalize the training targets.\n") ;		
		printf("  -a     activates loading weighted kernels.\n") ;
		printf("  -T  t  set Tolerance at t (default 0.001).\n") ;
		printf("  -K o set kappa value at o (default 1).\n") ;	
		printf("  -C o set C value at o (default  1).\n") ;	
		printf("\n") ;
		if (NULL !=defsetting)
			Clear_def_Settings( defsetting ) ;
		return 0;
	}
	else
	{
		if (argc>1)
			printf("Options:\n") ;
		do
		{
			strcpy(buf, argv[--argc]) ;
			sz = strlen(buf) ;
			//printf ("%s  %d\n", buf, sz) ;
			if ( '-' == buf[0] )
			{				
				for (index = 1 ; index < sz ; index++)
				{
					switch (buf[index])
					{
					case 'v' :
						printf("  - Verbose mode in display.\n") ;
						defsetting->smo_display = TRUE ;
						break ;	
					case 'a' :
						printf("  - kernels with Ard parameters.\n") ;
						defsetting->ardon = TRUE ;	
						break ;
					case 'o' :
						printf("  - normalize the Outputs in training data.\n") ;
						defsetting->normalized_output = TRUE ;	
						defsetting->pairs.normalized_output = TRUE ;
						break ;
					case 'i' :
						printf("  - normalize the Inputs in training data.\n") ;
						defsetting->normalized_input = TRUE ;	
						defsetting->pairs.normalized_input = TRUE ;
						break ;
					case 'L' :
						printf("  - choose Linear kernel.\n") ;
						defsetting->kernel = LINEAR ;						
						break ;
					case 'E' :
						if (parameter>0)
						{
							printf("  - set Epsilon as %.3f.\n", parameter) ;
							defsetting->epsilon = parameter ;
						}
						break ;					
					case 'T' :
						if (parameter>0)
						{
							printf("  - set Tol as %.6f.\n", parameter) ;
							defsetting->tol = parameter ;
						}
						break ;
					case 'C' :
						if (parameter > 0)
						{ 
							defsetting->vc = (parameter) ;
							printf("  - C at %f.\n", parameter) ;
							parameter = 0 ;					
						}
						
						break ;						
					case 'K' :
						if (parameter > 0)
						{ 
							defsetting->kappa = (parameter) ;
							printf("  - K at %f.\n", parameter) ;
							parameter = 0 ;						
						}
						break ;
					case 'P' :						
						if (parameter >= 1)
						{ 
							defsetting->kernel = POLYNOMIAL ;
							defsetting->p = (unsigned int) parameter ;
							printf("  - choose Polynomial kernel with order %d.\n", defsetting->p) ;
							parameter = 0 ;
						}					
						break ;	
					default :
						if ('-' != buf[index])
							printf("  -%c is invalid.\n", buf[index]) ;
						break ;
					}
				}
			}
			else
				parameter = atof(buf) ;
		}
		while ( argc > 1 ) ;
		printf("\n") ;
	}

	sprintf(filename,"%s",defsetting->inputfile) ;
	log = fopen ("validation_explicit.log", "w+t") ;
	if (NULL != log)
		fclose(log) ;	// clear the old file.

	while ( TRUE == Update_def_Settings(defsetting) ) 
	{
		defsetting->training.count = defsetting->pairs.count ;		
		defsetting->training.front = defsetting->pairs.front ;		
		defsetting->training.rear = defsetting->pairs.rear ;
		defsetting->training.classes = defsetting->pairs.classes ;	
		defsetting->training.dimen = defsetting->pairs.dimen ;
		defsetting->training.featuretype = defsetting->pairs.featuretype ;
		defsetting->training.datatype = defsetting->pairs.datatype ;
		// create smosettings
		printf ("\n\n TESTING on %s...\n", defsetting->testfile ) ;	
		smosetting = Create_smo_Settings(defsetting) ; 
		smosetting->pairs = &defsetting->pairs ;  		
		defsetting->training.count = 0 ;		
		defsetting->training.front = NULL ;		
		defsetting->training.rear = NULL ;
		defsetting->training.featuretype = NULL ;
		// load test data
		if ( FALSE == smo_Loadfile(&(defsetting->testdata), defsetting->testfile, defsetting->pairs.dimen) )
		{
			printf ("No testing data found in the file %s.\n", defsetting->testfile ) ;
			svm_saveresults (&defsetting->pairs, smosetting) ;		
		}
		// calculate the test output
		else
		{
			smo_routine (smosetting) ;
			svm_predict (&defsetting->testdata, smosetting) ;
			svm_saveresults (&defsetting->testdata, smosetting) ;

			if (REGRESSION == smosetting->pairs->datatype)
				printf("\r\nTEST ASE %f, AAE %f and SVs %.0f at C=%f and Kappa=%f with %.3f seconds.\n", smosetting->testrate, smosetting->testerror, smosetting->svs, smosetting->vc, smosetting->kappa, smosetting->smo_timing) ;
			else if (ORDINAL == smosetting->pairs->datatype)
				printf ("\r\nTEST ERROR NUMBER %.0f, AAE %.0f and SVs %.0f, at C=%.3f Kappa=%.3f with %.3f seconds.\n", 
				smosetting->testerror*defsetting->testdata.count,smosetting->testrate*defsetting->testdata.count, smosetting->svs, smosetting->vc, smosetting->kappa, smosetting->smo_timing) ;
			else
				printf ("\r\nTEST ERROR %f and SVs %.0f and Kappa=%f with %.3f seconds.\n", smosetting->testerror, smosetting->svs, smosetting->kappa, smosetting->smo_timing) ;

			if (NULL != (log = fopen ("kfoldsvc.log", "a+t")) ) 
			{
				if (REGRESSION == smosetting->pairs->datatype)
					fprintf(log,"TEST ASE %f, AAE %f and SVs %.0f at C=%f and Kappa=%f with %.3f seconds.\n", smosetting->testrate, smosetting->testerror, smosetting->svs, smosetting->vc, smosetting->kappa, smosetting->smo_timing) ;
				else if (ORDINAL == smosetting->pairs->datatype)
					fprintf (log,"TEST ERROR NUMBER %.0f, AAE %.0f and SVs %.0f, at C=%.3f Kappa=%.3f with %.3f seconds.\n", 
					smosetting->testerror*defsetting->testdata.count,smosetting->testrate*defsetting->testdata.count, smosetting->svs, smosetting->vc, smosetting->kappa, smosetting->smo_timing) ;
				else
					fprintf(log,"TEST ERROR %f and SVs %.0f at C=%f and Kappa=%f with %.3f seconds.\n", smosetting->testerror, smosetting->svs, smosetting->vc, smosetting->kappa, smosetting->smo_timing) ;		
				fclose(log) ;
			}
			// write another log				
			if (REGRESSION == smosetting->pairs->datatype)
			{
				if (NULL != (log = fopen ("esvr.log", "a+t")) )
				{
					fprintf(log,"%f %f\n", smosetting->testerror, smosetting->testrate) ;
					fclose(log) ;			
				}
			}
			else if (ORDINAL == smosetting->pairs->datatype)
			{
				if (NULL != (log = fopen ("ordinal_explicit.log", "a+t")) )
				{
					fprintf(log,"%.0f %.0f %f %f\n", smosetting->testerror*defsetting->testdata.count, smosetting->testrate*defsetting->testdata.count, smosetting->testrate, smosetting->smo_timing) ;
					fclose(log) ;                   
				}
			}
		}
		Clear_smo_Settings( smosetting ) ;
	}
	// free memory then exit
	Clear_def_Settings( defsetting ) ;	
	return 0;
}
//end of main.c 
