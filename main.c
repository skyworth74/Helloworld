#include "stdio.h"
int main(void)
{
    unsigned char i =0,j=0;
    unsigned char circlebuf[12]={1,2,3,4,5,6,7,8,9,0x0a,0x0b,0x0c};
    printf("helloworld\r\n");
    printf("----------------------------------------\r\n");
    printf("--                                    --\r\n");
    printf("--                                    --\r\n");
    printf("--                                    --\r\n");
    printf("--                                    --\r\n");
    printf("1\r\n");
    printf("----------------------------------------\r\n");
    printf("circle\r\n");

    for (i=0;i<10;i++){
         j=(j+1)%10;   
    }
    while(1){
        printf("0x%2x\r\n",circlebuf[i]);
        i=(i+1)%12;

    }
    return 1;
}
