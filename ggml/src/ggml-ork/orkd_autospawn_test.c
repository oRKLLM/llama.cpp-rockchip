/* auto-spawn test: no daemon is started manually; orkd_connect() must spawn one. arg1 = seconds to hold. */
#include "orkd_client.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
int main(int argc, char **argv){
    int hold = argc > 1 ? atoi(argv[1]) : 0;
    orkd_conn *c = orkd_connect();
    if (!c){ printf("[pid %d] connect FAILED\n", (int)getpid()); return 1; }
    printf("[pid %d] connected client_id=%u\n", (int)getpid(), orkd_client_id(c));
    if (orkd_ping(c)){ printf("[pid %d] ping FAILED\n", (int)getpid()); return 2; }
    printf("[pid %d] ping OK\n", (int)getpid());
    if (hold > 0) sleep(hold);
    orkd_disconnect(c);
    printf("[pid %d] disconnected\n", (int)getpid());
    return 0;
}
