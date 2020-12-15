#define PI 3.141592653589723234
#define RAD(deg) (deg * PI / 180)

#define IMG_SIZE 512
#define THRESHOLD 50

#if IMG_SIZE == 256
#define MAX_R 362
#define IMG_FILE "img/ipad256.jpg"
#elif IMG_SIZE == 512
#define MAX_R 724
#define IMG_FILE "img/testx.jpg"
#elif IMG_SIZE == 1024
#define MAX_R 1448
#define IMG_FILE "img/ipad1024.jpg"
#else
#define MAX_R 2896
#define IMG_FILE "img/ipad2048.jpg"
#endif

#define MIN_R -MAX_R
#define MAX_THETA RAD(89)
#define MIN_THETA RAD(-89)
#define THETA_STEPS MAX_R
#define D_THETA (MAX_THETA - MIN_THETA) / THETA_STEPS

// Macro to access array easier
#define GETIM(img, x, y) img[y * IMG_SIZE + x]
#define GETACC(acc, row, col) acc[row * THETA_STEPS + col]

// Bound check
#define IN_BOUND(x, y) (x < IMG_SIZE && x >= 0 && y < IMG_SIZE && y >= 0)
