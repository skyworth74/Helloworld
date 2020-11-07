#include "stdio.h"
int main(void)
{
    unsigned char i =0,j=0;
    printf("helloworld\r\n");
    printf("----------------------------------------\r\n");
    printf("--                                    --\r\n");
    printf("--                                    --\r\n");
    printf("--                                    --\r\n");
    printf("--                                    --\r\n");
    printf("1\r\n");
    printf("----------------------------------------\r\n");
    for (i=0;i<10;i++){
         j=(j+1)%10;   
    }
    return 1;
}
