				/************************/
				/*   Table of 		*/	
				/*   Sine Function	*/
				/************************/
				
				/* Michel Vallieres 	*/
				/* Written: Winter 1995	*/
#include <stdio.h>
#include <math.h>

int main()
{
    int    angle_degree;
    double angle_radian, value;

					/* Print a header */
    printf ("\nCompute a table of the sine function\n\n");

    printf ( " angle     Sine \n" );

    angle_degree=0;			/* initial angle value 		 */
					/* scan over angle 		 */

    while (  angle_degree <= 360 )	/* loop until angle_degree > 360 */
    {
       angle_radian = M_PI * angle_degree/180.0 ;
       value = sin(angle_radian);
       printf ( " %3d      %f \n ", angle_degree, value );

       angle_degree = angle_degree + 10; /* increment the loop index	 */
    }
}

