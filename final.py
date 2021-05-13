from mymath import MSE
from mymath import PSNR
from mymath import SSIM
import cv2

def calculator():
    num = 1
    sumM1 = 0
    sumS1 = 0
    sumP1 = 0
    sumM2 = 0
    sumS2 = 0
    sumP2 = 0

    for i in range(500, 49501, 500):
        a = eval('str(i)+"_1_ImageNet_input.png"')
        b = eval('str(i)+"_0.0_1_24000_ImageNet_output.png"')
        c = eval('str(i)+"_0.5_1_24000_ImageNet_output.png"')
        
        original = cv2.imread(a)
        noprune = cv2.imread(b)
        prune =  cv2.imread(c)

        M1 = MSE(original, noprune)
        S1 = SSIM(original, noprune)
        P1 = PSNR(original, noprune)
        
        M2 = MSE(original, prune)
        S2 = SSIM(original, prune)
        P2 = PSNR(original, prune)

        num += 1
        sumM1 += M1
        sumS1 += S1
        sumP1 += P1 
        sumM2 += M2
        sumS2 += S2
        sumP2 += P2

    return num, sumM1, sumS1, sumP1, sumM2, sumS2, sumP2

num, sumM1, sumS1, sumP1, sumM2, sumS2, sumP2 = calculator()
print(num)
print(sumM1/num)
print(sumS1/num)
print(sumP1/num)
print(sumM2/num)
print(sumS2/num)
print(sumP2/num)
