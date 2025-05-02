import numpy as np
import cv2
import matplotlib.pyplot as plt

def calc():
    # numpyで行列の四則演算
    x = np.array([[5,10],[15,20]])
    y = np.array([[1,2],[3,4]])

    add = x + y
    sub = x - y
    mul = x * y
    div = x / y
    dot = np.dot(x, y)

    print("addition:\n{}".format(add))
    print("subtraction:\n{}".format(sub))
    print("multiplication:\n{}".format(mul))
    print("division:\n{}".format(div))
    print("dot product:\n{}".format(dot))


def image():
    # 画像の表示、縮小拡大、回転、二値化

    im = cv2.imread('girl.png')
    cv2.imwrite("a.png", im)
    print(im.shape)
    h, w = im.shape[:2]

    # 表示
    cv2.imshow('girl', im)
    cv2.waitKey(2000)

    # 縮小
    im_small = cv2.resize(im,(int(w/4), int(h/2)))
    print("small_shape:{}".format(im_small.shape))
    cv2.imwrite("small.png", im_small)
    #cv2.imshow('girl_small', im_small)
    #cv2.waitKey(2000)

    # 拡大
    im_big = cv2.resize(im,(w*4, h*2))
    print("big_shape:{}".format(im_big.shape))
    cv2.imwrite("big.png", im_big)
    #cv2.imshow('girl_big', im_big)
    #cv2.waitKey(2000)

    fig, axes = plt.subplots(1, 2, figsize=(10, 10))
    axes[0].imshow(im_big)
    axes[0].set_title("Big Image")
    axes[0].axis("off")
    axes[1].imshow(im_small)
    axes[1].set_title("Small Image")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("big_small.png")

    # 回転
    im_route = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
    print("route_shape:{}".format(im_route.shape))
    cv2.imwrite("route.png", im_route)

    # 二値化
    im2 = cv2.imread("girl.png", 0)
    print(im2.shape)
    thr, img_th = cv2.threshold(im2, 100, 255, cv2.THRESH_BINARY) # thr:閾値、img_th:二値化画像
    print("thr:{}".format(thr))
    cv2.imwrite("thr.png", img_th)

    print("done")


def difference():

    im = cv2.imread('girl.png')
    h, w = im.shape[:2]
    
    # 直線を描画
    im_line = cv2.line(im, (int(h/4), int(w/2)), (int(h/2), int(w/2)), (255,0,0), 5)
    #cv2.imshow('line',im_line)
    #cv2.waitKey(2000)

    # 直線なしを読み直し
    im = cv2.imread('girl.png')

    # 差分
    diff = cv2.absdiff(im, im_line)

    # グレースケール
    gray2 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # BGRからRGBに変換 imshowはBGR
    image1_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(im_line, cv2.COLOR_BGR2RGB)

    # 差分画像を正規化（カラーマップを適用するため）
    norm_diff = gray_diff / np.max(gray_diff)

    # 差分画像に重みをかけて2枚目の画像の色に反映
    diff_img = cv2.addWeighted(gray2, 0.1, gray_diff, 2, 100)

    diff_colored = np.zeros_like(image2_rgb)
    diff_colored[..., 0] = image2_rgb[..., 0] * norm_diff
    diff_colored[..., 1] = image2_rgb[..., 1] * norm_diff
    diff_colored[..., 2] = image2_rgb[..., 2] * norm_diff

    # 結果をMatplotlibで表示
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # 元画像
    axes[0, 0].imshow(image1_rgb)
    axes[0, 0].set_title("Image 1")
    axes[0, 0].axis("off")

    # 直線画像
    axes[0, 1].imshow(image2_rgb)
    axes[0, 1].set_title("Image 2")
    axes[0, 1].axis("off")

    # 差分画像（グレースケール）
    axes[1, 0].imshow(diff_img, cmap="gray")
    axes[1, 0].set_title("Difference (Grayscale)")
    axes[1, 0].axis("off")

    # 差分画像（カラー）
    axes[1, 1].imshow(diff_colored)
    axes[1, 1].set_title("Difference (Colored)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig("diff.png")
    #plt.show()


def features():

    im = cv2.imread('cat.png', 0)
    im2 = cv2.imread('airplain.png', 0)
    
    # ヒストグラムの作成
    hist = cv2.calcHist([im], [0], None, [256], [0,256])
    hist2 = cv2.calcHist([im2], [0], None, [256], [0,256])

    plt.plot(hist)
    plt.savefig("hist_cat.png")
    plt.clf()

    plt.plot(hist2)
    plt.savefig("hist_airplain.png")

    
    # sift
    im = cv2.imread('cat.png',0)
    sift = cv2.xfeatures2d.SIFT_create()
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = im
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    img_sift = cv2.drawKeypoints(gray, keypoints, None, flags=4)
    cv2.imwrite("sift_cat.jpg",img_sift)

    im2 = cv2.imread('airplain.png',0)
    sift = cv2.xfeatures2d.SIFT_create()
    gray = im2
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    img_sift = cv2.drawKeypoints(gray, keypoints, None, flags=4)
    cv2.imwrite("sift_airplain.jpg",img_sift)


    # akaze
    im = cv2.imread('cat.png')
    gray1 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    akaze = cv2.AKAZE_create()
    kp1 = akaze.detect(gray1)
    img1_akaze = cv2.drawKeypoints(gray1, kp1, None, flags=4)
    cv2.imwrite("akaze_cat.jpg",img1_akaze)

    im2 = cv2.imread('airplain.png')
    gray1 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    akaze = cv2.AKAZE_create()
    kp1 = akaze.detect(gray1)
    img1_akaze = cv2.drawKeypoints(gray1, kp1, None, flags=4)
    cv2.imwrite("akaze_airplain.jpg",img1_akaze)



calc()
image()
difference()
features()