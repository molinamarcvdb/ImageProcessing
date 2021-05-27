function MAM_SEG3
%% INICIALITZACIÓ
clearvars; close all; clc;
%% Read and gray
Img=imread('images/Original_3_b3.jpg');
figure(1); imshow(Img); title('Original Image');
frm = 0;
% Hi trobem molt soroll del tipus salt & pepper per el que aplicarem un
% filtre de mediana per reduirne el efecte 
%% Extreure soroll 
Img=medfilt2(Img,[4 4]);  
figure(2);
imagesc(Img,[0, 255]); axis off; axis equal; colormap(gray);
title('Smoothed imatge');
%% Binaritzar 
Img_b= imbinarize(Img)
figure(3);
imshow(Img_b); title('Imatge binaritzada');
%% Extreure elements petits
Img_bw = bwareaopen(Img_b, 200)
SE=strel('sphere', 20)
Mask=im2uint8(imdilate(Img_bw, SE));
figure(4); imshow(Mask); title('Mascara sense Etiqueta');
%% Convolucionem la imatge normal amb la imatge binaria sense etiqueta
mask = im2uint8(Mask)
maskedImage = Img;
maskedImage(~mask) = 0;
%Img_c = Img.*Img_bw
figure(5); imshow(maskedImage); title('Conv. Imatge-Màscara');
%% Gaussian
G=fspecial('gaussian',30,6); % 15 Caussian kernel
Img_G=conv2(maskedImage,G,'same');  % smooth image by Gaussiin convolution
figure(6);
imagesc(Img_G,[0, 255]); axis off; axis equal; colormap(gray);
title('Blurred image');
%% Extracció de gradients
[Ix,Iy]=gradient(Img_G);
f=Ix.^2+Iy.^2;
g=1./(1+f);  % edge indicator function.
g=exp(-f);
g=medfilt2(g,[5 5]);  

figure(7); subplot(1,2,1);imshow(f); axis off; axis equal; title('Imatge Gradient');
subplot(1,2,2); imshow(g); axis off; axis equal; title('Imatge Gradient Invers');
%% Paràmetres necessaris
% Un cop extret els gradients utilitzarem el mètode DRLSE utilitzat en la
% pràctica de segementació. Per inicialitzarlo primer hem d'iniciar els
% valors d'uns paràmetres que caracteritzarant el funcionament del
% algorisme.
timestep=1;  % time step
mu=0.2;  % coefficient of the distance regularization term R(phi)
lambda=5; %coefficient of the weighted length term L(phi)
alfa= -3;  %  coefficient of the weighted area term A(phi)
epsilon=2.5; % papramater that specifies the width of the DiracDelta function
c0=3;
maxiter=1501;
sigma=3.0;    % scale parameter in Gaussian kernel

%% Set initial phi
phi = c0*ones(size(maskedImage));
phi(320:362,10:20)=-c0;
phi(1:60,200:235)=-c0;
phi(1:60,1:60)=-c0;
figure(8);
imagesc(phi);
axis off; axis equal;colormap(jet);
title('initial phi matrix');

[vx, vy]=gradient(g);
figure(9);
subplot(1,2,1);imagesc(vx); title('x directioned gradient of g');
subplot(1,2,2);imagesc(vy); title('y directioned gradient of g');

for k=1:maxiter
    %% step6, check boundary conditions
    phi=NeumannBoundCond(phi);
    
    %% step 7 calculate differential of regularized term in Eq.30
    distRegTerm=distReg_p2(phi);
    
    %% step8 calculate differential of area term in Eq.30
    diracPhi=Dirac(phi,epsilon);
    areaTerm=diracPhi.*g;
    
    %% step9 calculate differential of length term in Eq.30
    [phi_x,phi_y]=gradient(phi);
    s=sqrt(phi_x.^2 + phi_y.^2);
    Nx=phi_x./(s+1e-10); % add a small positive number to avoid division by zero
    Ny=phi_y./(s+1e-10);
    edgeTerm=diracPhi.*(vx.*Nx+vy.*Ny) + diracPhi.*g.*div(Nx,Ny);
    
    %% step 10 update phi according to Eq.20
    phi=phi + timestep*(mu/timestep*distRegTerm + lambda*edgeTerm + alfa*areaTerm);
    
    %% show result in every 50 iteration
    if mod(k,50)==1
        frm=frm+1;
        h=figure(10);
        set(gcf,'color','w');
        subplot(1,2,1);
        II=maskedImage;
        II(:,:,2)=maskedImage;II(:,:,3)=maskedImage;
        imshow(II); axis off; axis equal; hold on;  
        q=contour(phi, [0,0], 'r');
        msg=['contour result , iteration number=' num2str(k)];
        title(msg);
        subplot(1,2,2);
        mesh(-phi); 
        hold on;  contour(phi, [0,0], 'r','LineWidth',2);
        
        view([-55+180 55]);      
        msg=['phi result , iteration number=' num2str(k)];
        title(msg);
        pause(0.1)
        
    frame = getframe(h);
        im = frame2im(frame);
        [imind,cm] = rgb2ind(im,256);
        %Write to the GIF File
        if frm == 1        
            imwrite(imind,cm,'MAMseg3.gif','gif', 'Loopcount',inf);
        else        
            imwrite(imind,cm,'MAMseg3.gif','gif','WriteMode','append');
        end
    
    end
    
    
    %% step 11 if maxiter done then finish, else return step6
end
%% Step 12. show last iteration results
figure(11);
imagesc(Img,[0, 255]); axis off; axis equal; colormap(gray); hold on;  contour(phi, [0,0], 'r');
msg=['phi result , iteration number=' num2str(k)];
title(msg);

%% Convolucionem les dos mascares, seguit d'una erosió i una obertura
mask = Mask;
mask2 = im2uint8(phi);
mask= mask.*mask2;
SE4 = strel('diamond',10);
mask = imerode( mask, SE4);
SE3 = strel('sphere',20);
mask = imopen(mask, SE3);
SE5 = strel('octagon',6);
mask= imdilate(mask, SE5);


Im_mask =imadjust(maskedImage);
Im_mask(~mask) = 0;
Im_mask= localcontrast(Im_mask);
figure(12);subplot(131);imshow(Img); title('Original Image'); axis off; hold on;
subplot(132);imshow(mask); title('Màscara obtinguda'); axis off; hold on;
subplot(133); imshow(Im_mask);title('ROI'); axis off; hold off

function f = distReg_p2(phi)
% compute the distance regularization term with the double-well potential p2 in eqaution (16)
[phi_x,phi_y]=gradient(phi);
s=sqrt(phi_x.^2 + phi_y.^2);
a=(s>=0) & (s<=1);
b=(s>1);
ps=a.*sin(2*pi*s)/(2*pi)+b.*(s-1);  % compute first order derivative of the double-well potential p2 in eqaution (16)
dps=((ps~=0).*ps+(ps==0))./((s~=0).*s+(s==0));  % compute d_p(s)=p'(s)/s in equation (10). As s-->0, we have d_p(s)-->1 according to equation (18)
f = div(dps.*phi_x - phi_x, dps.*phi_y - phi_y) + 4*del2(phi);

function f = div(nx,ny)
[nxx,junk]=gradient(nx);
[junk,nyy]=gradient(ny);
f=nxx+nyy;

function f = Dirac(x, sigma)
f=(1/2/sigma)*(1+cos(pi*x/sigma));
b = (x<=sigma) & (x>=-sigma);
f = f.*b;

function g = NeumannBoundCond(f)
% Make a function satisfy Neumann boundary condition
[nrow,ncol] = size(f);
g = f;
g([1 nrow],[1 ncol]) = g([3 nrow-2],[3 ncol-2]);
g([1 nrow],2:end-1) = g([3 nrow-2],2:end-1);
g(2:end-1,[1 ncol]) = g(2:end-1,[3 ncol-2]);



