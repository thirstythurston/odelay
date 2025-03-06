import cv2
import numpy as np


#This function measures the relative degree of focus of 
#an image. It may be invoked as:
#
#   FM = fmeasure(Image, Method, ROI)
#
#Where 
#   Image,  is a grayscale image and FM is the computed
#           focus value.
#   Method, is the focus measure algorithm as a string.
#           see 'operators.txt' for a list of focus 
#           measure methods. 
#   ROI,    Image ROI as a rectangle [xo yo width heigth].
#           if an empty argument is passed, the whole
#           image is processed.
#
#  Said Pertuz
#  Abr/2010



def fmeasure(Iamge, Measure, ROI):

    if ~isempty(ROI)
        Image = imcrop(Image, ROI);
    

    WSize = 15; # Size of local window (only some operators)

    Measure = upper(Measure)
    if Measure == 'ACMO': # Absolute Central Moment (Shirvaikar2004)
        if dtype(Image) is not int:
            Image = Image.astype('uint8');
        
        FM = AcMomentum(Image);
                    
        elif Measure == 'BREN': # Brenner's (Santos97)
            [M N] = Image.shape
            DH = Image.copy
            DV = Image.copy
            DH[:M-2,:] = np.diff(Image, n=2, axis=1)
            DV[:,:N-2] = np.diff(Image, n=2, axis=2)
            FM = np.maximum(DH, DV)
            FM = FM**2
            FM = np.mean(FM)
            
        elif Measure == 'CONT': # Image contrast (Nanda2001)
            
            FM = cv2.medianBlur(Image, [3 3])
            FM = np.mean(FM)
                            
        # elif Measure == 'CURV': # Image Curvature (Helmli2001)
        #     if ~isinteger(Image), Image = im2uint8(Image);
            
        #     M1 = np.array(([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]]), dtype = 'uint16')
        #     M2 = [1 0 1;1 0 1;1 0 1];
        #     P0 = imfilter(Image, M1, 'replicate', 'conv')/6;
        #     P1 = imfilter(Image, M1.T, 'replicate', 'conv')/6;
        #     P2 = 3*imfilter(Image, M2, 'replicate', 'conv')/10 ...
        #         -imfilter(Image, M2.T, 'replicate', 'conv')/5;
        #     P3 = -imfilter(Image, M2, 'replicate', 'conv')/5 ...
        #         +3*imfilter(Image, M2, 'replicate', 'conv')/10;
        #     FM = abs(P0) + abs(P1) + abs(P2) + abs(P3);
        #     FM = np.mean(FM);
            
        # elif Measure == 'DCTE' # DCT energy ratio (Shen2006)
        #     FM = nlfilter(Image, [8 8], @DctRatio);
        #     FM = np.mean(FM);
            
        # elif Measure == 'DCTR' # DCT reduced energy ratio (Lee2009)
        #     FM = nlfilter(Image, [8 8], @ReRatio);
        #     FM = np.mean(FM);
            
        # elif Measure == 'GDER' # Gaussian derivative (Geusebroek2000)        
        #     N = floor(WSize/2);
        #     sig = N/2.5;
        #     [x,y] = meshgrid(-N:N, -N:N);
        #     G = exp(-(x.^2+y.^2)/(2*sig^2))/(2*pi*sig);
        #     Gx = -x.*G/(sig^2);Gx = Gx/sum(Gx(:));
        #     Gy = -y.*G/(sig^2);Gy = Gy/sum(Gy(:));
        #     Rx = imfilter(double(Image), Gx, 'conv', 'replicate');
        #     Ry = imfilter(double(Image), Gy, 'conv', 'replicate');
        #     FM = Rx.^2+Ry.^2;
        #     FM = np.mean(FM);
            
        # elif Measure == 'GLVA' # Graylevel variance (Krotkov86)
        #     FM = std2(Image);
            
        # elif Measure == 'GLLV' #Graylevel local variance (Pech2000)        
        #     LVar = stdfilt(Image, ones(WSize,WSize)).^2;
        #     FM = std2(LVar)^2;
            
        # elif Measure == 'GLVN' # Normalized GLV (Santos97)
        #     FM = std2(Image)^2/np.mean(Image);
            
        # elif Measure == 'GRAE' # Energy of gradient (Subbarao92a)
        #     Ix = Image;
        #     Iy = Image;
        #     Iy(1:-1,:) = diff(Image, 1, 1);
        #     Ix(:,1:-1) = diff(Image, 1, 2);
        #     FM = Ix.^2 + Iy.^2;
        #     FM = np.mean(FM);
            
        # elif Measure == 'GRAT' # Thresholded gradient (Snatos97)
        #     Th = 0; #Threshold
        #     Ix = Image;
        #     Iy = Image;
        #     Iy(1:-1,:) = diff(Image, 1, 1);
        #     Ix(:,1:-1) = diff(Image, 1, 2);
        #     FM = max(abs(Ix), abs(Iy));
        #     FM(FM<Th)=0;
        #     FM = sum(FM(:))/sum(sum(FM~=0));
            
        # elif Measure == 'GRAS' # Squared gradient (Eskicioglu95)
        #     Ix = diff(Image, 1, 2);
        #     FM = Ix.^2;
        #     FM = np.mean(FM);
            
        # elif Measure == 'HELM' #Helmli's mean method (Helmli2001)        
        #     MEANF = fspecial('average',[WSize WSize]);
        #     U = imfilter(Image, MEANF, 'replicate');
        #     R1 = U./Image;
        #     R1(Image==0)=1;
        #     index = (U>Image);
        #     FM = 1./R1;
        #     FM(index) = R1(index);
        #     FM = np.mean(FM);
            
        # elif Measure == 'HISE' # Histogram entropy (Krotkov86)
        #     FM = entropy(Image);
            
        # elif Measure == 'HISR' # Histogram range (Firestone91)
        #     FM = max(Image(:))-min(Image(:));
            
            
        # elif Measure == 'LAPE' # Energy of laplacian (Subbarao92a)
        #     LAP = fspecial('laplacian');
        #     FM = imfilter(Image, LAP, 'replicate', 'conv');
        #     FM = np.mean(FM.^2);
                    
        # elif Measure == 'LAPM' # Modified Laplacian (Nayar89)
        #     M = [-1 2 -1];        
        #     Lx = imfilter(Image, M, 'replicate', 'conv');
        #     Ly = imfilter(Image, M', 'replicate', 'conv');
        #     FM = abs(Lx) + abs(Ly);
        #     FM = np.mean(FM);
            
        elif Measure == 'LAPV' # Variance of laplacian (Pech2000)
            
            ILAP = cv2.Laplacian(Image,cv2.CV_64F ).var()
            FM = np.std(ILAP)**2
            
        # elif Measure == 'LAPD' # Diagonal laplacian (Thelen2009)
        #     M1 = [-1 2 -1];
        #     M2 = [0 0 -1;0 2 0;-1 0 0]/sqrt(2);
        #     M3 = [-1 0 0;0 2 0;0 0 -1]/sqrt(2);
        #     F1 = imfilter(Image, M1, 'replicate', 'conv');
        #     F2 = imfilter(Image, M2, 'replicate', 'conv');
        #     F3 = imfilter(Image, M3, 'replicate', 'conv');
        #     F4 = imfilter(Image, M1', 'replicate', 'conv');
        #     FM = abs(F1) + abs(F2) + abs(F3) + abs(F4);
        #     FM = np.mean(FM);
            
        # elif Measure == 'SFIL' #Steerable filters (Minhas2009)
        #     # Angles = [0 45 90 135 180 225 270 315];
        #     N = floor(WSize/2);
        #     sig = N/2.5;
        #     [x,y] = meshgrid(-N:N, -N:N);
        #     G = exp(-(x.^2+y.^2)/(2*sig^2))/(2*pi*sig);
        #     Gx = -x.*G/(sig^2);Gx = Gx/sum(Gx(:));
        #     Gy = -y.*G/(sig^2);Gy = Gy/sum(Gy(:));
        #     R(:,:,1) = imfilter(double(Image), Gx, 'conv', 'replicate');
        #     R(:,:,2) = imfilter(double(Image), Gy, 'conv', 'replicate');
        #     R(:,:,3) = cosd(45)*R(:,:,1)+sind(45)*R(:,:,2);
        #     R(:,:,4) = cosd(135)*R(:,:,1)+sind(135)*R(:,:,2);
        #     R(:,:,5) = cosd(180)*R(:,:,1)+sind(180)*R(:,:,2);
        #     R(:,:,6) = cosd(225)*R(:,:,1)+sind(225)*R(:,:,2);
        #     R(:,:,7) = cosd(270)*R(:,:,1)+sind(270)*R(:,:,2);
        #     R(:,:,7) = cosd(315)*R(:,:,1)+sind(315)*R(:,:,2);
        #     FM = max(R,[],3);
        #     FM = np.mean(FM);
            
        # elif Measure == 'SFRQ' # Spatial frequency (Eskicioglu95)
        #     Ix = Image;
        #     Iy = Image;
        #     Ix(:,1:-1) = diff(Image, 1, 2);
        #     Iy(1:-1,:) = diff(Image, 1, 1);
        #     FM = np.mean(sqrt(double(Iy.^2+Ix.^2)));
            
        # elif Measure == 'TENG'# Tenengrad (Krotkov86)
        #     Sx = fspecial('sobel');
        #     Gx = imfilter(double(Image), Sx, 'replicate', 'conv');
        #     Gy = imfilter(double(Image), Sx', 'replicate', 'conv');
        #     FM = Gx.^2 + Gy.^2;
        #     FM = np.mean(FM);
            
        # elif Measure == 'TENV' # Tenengrad variance (Pech2000)
        #     Sx = fspecial('sobel');
        #     Gx = imfilter(double(Image), Sx, 'replicate', 'conv');
        #     Gy = imfilter(double(Image), Sx', 'replicate', 'conv');
        #     G = Gx.^2 + Gy.^2;
        #     FM = std2(G)^2;
            
        elif Measure == 'VOLA' # Vollath's correlation (Santos97)
            Image = double(Image);
            I1 = Image; I1(1:-1,:) = Image(2:,:);
            I2 = Image; I2(1:-2,:) = Image(3:,:);
            Image = Image.*(I1-I2);
            FM = np.mean(Image);
            
        # elif Measure == 'WAVS' #Sum of Wavelet coeffs (Yang2003)
        #     [C,S] = wavedec2(Image, 1, 'db6');
        #     H = wrcoef2('h', C, S, 'db6', 1);   
        #     V = wrcoef2('v', C, S, 'db6', 1);   
        #     D = wrcoef2('d', C, S, 'db6', 1);   
        #     FM = abs(H) + abs(V) + abs(D);
        #     FM = np.mean(FM);
            
        # elif Measure == 'WAVV' #Variance of  Wav...(Yang2003)
        #     [C,S] = wavedec2(Image, 1, 'db6');
        #     H = abs(wrcoef2('h', C, S, 'db6', 1));
        #     V = abs(wrcoef2('v', C, S, 'db6', 1));
        #     D = abs(wrcoef2('d', C, S, 'db6', 1));
        #     FM = std2(H)^2+std2(V)+std2(D);
            
        # elif Measure == 'WAVR'
        #     [C,S] = wavedec2(Image, 3, 'db6');
        #     H = abs(wrcoef2('h', C, S, 'db6', 1));   
        #     V = abs(wrcoef2('v', C, S, 'db6', 1));   
        #     D = abs(wrcoef2('d', C, S, 'db6', 1)); 
        #     A1 = abs(wrcoef2('a', C, S, 'db6', 1));
        #     A2 = abs(wrcoef2('a', C, S, 'db6', 2));
        #     A3 = abs(wrcoef2('a', C, S, 'db6', 3));
        #     A = A1 + A2 + A3;
        #     WH = H.^2 + V.^2 + D.^2;
        #     WH = np.mean(WH);
        #     WL = np.mean(A);
        #     FM = WH/WL;
        # otherwise
        #     error('Unknown measure #s',upper(Measure))
    
    
#************************************************************************
function fm = AcMomentum(Image)
[M N] = size(Image);
Hist = imhist(Image)/(M*N);
Hist = abs((0:255)-255*np.mean(Image))'.*Hist;
fm = sum(Hist);


#******************************************************************
function fm = DctRatio(M)
MT = dct2(M).^2;
fm = (sum(MT(:))-MT(1,1))/MT(1,1);


#************************************************************************
function fm = ReRatio(M)
M = dct2(M);
fm = (M(1,2)^2+M(1,3)^2+M(2,1)^2+M(2,2)^2+M(3,1)^2)/(M(1,1)^2);

#******************************************************************
