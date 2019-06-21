function [out] = paper_demo

addpath(genpath('./UTILS'));

%% Setting parameters
global alpha alpha0 alpha1 lambda beta betaTVW b m n D R W;
maxit = 50;
sigmaTGVW = 1/sqrt(19);
tauTGVW = sigmaTGVW;
sigmaTGV = 1/sqrt(12);
tauTGV = sigmaTGV;
sigmaTVW = 1/sqrt(10);
tauTVW = sigmaTVW;
lambda = 8e-5;
alpha0 = 2;
alpha1 = 1;
beta =  0.75 ;
alpha = 0.001;
betaTVW = 0.035;
n = 2*128;
m = n;

%% Loadibg data
x0 = double(imread('knee.png')); % Load another image if needed 
size(x0)
x0 = imresize(x0, [n m]);
size(x0)
figure(1);imshow(uint8(x0));title('Ground Truth');
c = norm(x0,'fro');
load('random16_256.mat');             % Load another mask if needed
mask = Q1;
figure(2);imshow(mask);title('Sampling Mask');
imwrite(mask,'SM.png', 'png');
b = mask.*fft2c(x0);
zf = ifft2c(b);
figure(3);imshow(uint8(abs(zf)));title('ZF');
imwrite(uint8(zf),'ZF.png', 'png');

%% Defining operators
R =@(x) mask.*fft2c(x);
R_adj = @(r) ifft2c(mask.*r);
prox2_sigma = @(r,lambda) r/(lambda*sigmaTGVW+1);
mysnr = @(x,x0) 20*log10(norm(x0,'fro')/norm(x-x0,'fro')); 
LoG_filter = fspecial('log',[15 15], 1.5);
LoG0 = imfilter(x0, LoG_filter,'symmetric', 'conv'); normLoG0 = norm(LoG0,'fro');
HFEN = @(u) norm(imfilter(u, LoG_filter, 'symmetric', 'conv') - LoG0,'fro')/normLoG0;
D = @(u) cat(3,dxp(u),dyp(u));
div_1 = @(p) dxm(p(:,:,1)) + dym(p(:,:,2)); 
W = Wavelet('Daubechies',4,4);


%% solution by TV+W 
u =zf;
u_tild = zeros(m,n);
p = D(u);
r = zeros(m,n);
s = zeros(m,n);
counter=0;
SNR_TVW = zeros(maxit,1);
SSIM_TVW = zeros(maxit,1);
PE_TVW = zeros(maxit,1);
HFENvec_TVW = zeros(maxit,1);
%%% Main Iterations
tic;
 for j=1:maxit
     counter = counter+1;
    p = projP(p + sigmaTVW*(D(u_tild)),alpha);
    u_=u_tild;
    w_=W*u_tild;
    norm(u_-w_,2)
    s = max(min(s + sigmaTVW*(W*u_tild), betaTVW), -betaTVW); 
    r = prox2_sigma (r + sigmaTVW*(R(u_tild) - b),1);
    u_old = u;
    u = projC(u + tauTVW*(div_1(p) - R_adj(r) - W'*s),c);
    u = max(0,real(u));
    u_tild = 2*u - u_old;
    
    x = u;
    SNR_TVW(counter)=mysnr(abs(x),abs(x0));
    HFENvec_TVW(counter)= HFEN(abs(x));
    SSIM_TVW(counter)=ssim(abs(x),abs(x0));
    PE_TVW(counter) = objTVW(x);
    fprintf('iteration= %d   TVWenergy= %.2d   SNR= %.2f   SSIM= %.4f   HFEN= %.4f\n',...
    counter,PE_TVW(counter),SNR_TVW(counter),SSIM_TVW(counter),HFENvec_TVW(counter));
end
time_TVW = toc;

figure(4);imshow(uint8(u));title('TVW');
imwrite(uint8(u),'TVW.png', 'png');

disparityRange = [0 0.3];
figure(5)
imshow(abs(u-x0)/255,disparityRange);
title('TVW Error Map');
colormap(gca,jet) 
colorbar


%% solution by TGV (TGV-MRI)
u = zf;        
u_tild = zeros(m,n);
v = D(u);
v_tild = zeros(m,n,2);
p = zeros(m,n,2);
q = zeros(m,n,3);
r = zeros(m,n);
counter=0;
SNR_TGV = zeros(maxit,1);
SSIM_TGV = zeros(maxit,1);
PE_TGV = zeros(maxit,1);
HFENvec_TGV = zeros(maxit,1);

% Main iterations
tic;
 for j=1:maxit
    counter = counter + 1;
    p = projP(p + sigmaTGV*(D(u_tild)-v_tild),alpha1);
    q = projQ(q + sigmaTGV*E(v_tild),alpha0);
    r = prox2_sigma(r + sigmaTGV*(R(u_tild) - b),lambda);
    u_old = u;
    u = projC(u + tauTGV*(div_1(p) - R_adj(r)),c);
    u = max(0,real(u));
    u_tild = 2*u - u_old;
    v_old = v;
    v = v + tauTGV*(p + div_2(q));
    v_tild = 2*v - v_old;

    SNR_TGV(counter)=mysnr(abs(u),abs(x0));
    HFENvec_TGV(counter)= HFEN(real(u));
    SSIM_TGV(counter)=ssim(abs(u),abs(x0));
    PE_TGV(counter) = objTGV(u,v);
    fprintf('iteration= %d   TGVenergy= %.2d   SNR= %.2f   SSIM= %.4f   HFEN= %.4f\n',...
    counter,PE_TGV(counter),SNR_TGV(counter),SSIM_TGV(counter),HFENvec_TGV(counter));
 end
time_TGV = toc;

figure(6);imshow(uint8(u));title('TGV');
imwrite(uint8(u),'TGV.png', 'png');

disparityRange = [0 0.3];
figure(7)
imshow(abs(u-x0)/255,disparityRange);
title('TGV Error map');
colormap(gca,jet) 
colorbar

%% solution by proposed method (TGVW)
u = zf;   
u_tild = zeros(m,n);
v = D(u);
v_tild = zeros(m,n,2);
p = zeros(m,n,2);
q = zeros(m,n,3);
r = zeros(m,n);
s = zeros(m,n);
SNR_TGVW = zeros(maxit,1);
SSIM_TGVW = zeros(maxit,1);
PE_TGVW = zeros(maxit,1);
HFENvec_TGVW = zeros(maxit,1);
counter=0;

% Main iterations
figure(1);
tic;
for mm = 1:maxit
counter = counter+1;
    p = projP(p + sigmaTGVW*(D(u_tild)-v_tild),alpha1);
    q = projQ(q + sigmaTGVW*E(v_tild),alpha0);
    s = max(min(s + sigmaTGVW*(W*u_tild), beta), -beta); 
    r = prox2_sigma (r + sigmaTGVW*(R(u_tild) - b),lambda);
    u_old = u;
    u = projC(u + tauTGVW*(div_1(p) - R_adj(r) - W'*s),c);         
    u = max(0,real(u));
    u_tild = 2*u - u_old;
    v_old = v;
    v = v + tauTGVW*(p + div_2(q));
    v_tild = 2*v - v_old;
   
    SNR_TGVW(counter)=mysnr( real(u),real (x0));
    HFENvec_TGVW(counter)= HFEN(real(u));
    SSIM_TGVW(counter)=ssim( real(u), real(x0));
    PE_TGVW(counter) = objTGVW(u,v);
    fprintf('iteration= %d   TGVWenergy= %.2d   SNR= %.2f   SSIM= %.4f   HFEN= %.4f\n',...
    counter,PE_TGVW(counter),SNR_TGVW(counter),SSIM_TGVW(counter),HFENvec_TGVW(counter));
end
time_TGVW = toc;

figure(8);imshow(uint8(u));title('Proposed');
imwrite(uint8(u),'TGVW.png', 'png');

disparityRange = [0 0.3];
figure(9)
imshow(abs(u-x0)/255,disparityRange);
title('Proposed Method Error Map');
colormap(gca,jet) 
colorbar

%% Graph of SNR
SNRzf = mysnr(abs(zf),abs(x0))*ones(maxit,1);
figure(10); clf;
h=plot(1:1:counter,SNR_TGVW,'k'); 
set(h,'LineWidth',2);
hold on;
h=plot(1:1:counter,SNR_TGV,'-.'); 
set(h,'LineWidth',2);
hold on;
h=plot(1:1:counter,SNR_TVW,':'); 
set(h,'LineWidth',2);
hold on;
h=plot(1:1:counter,SNRzf,'--'); 
set(h,'LineWidth',2);
set(gca,'FontSize',14);
h=xlabel('Iterations');
set(h,'FontSize',14);
h=ylabel('SNR');
set(h,'FontSize',14);
legend({'Proposed','TGV','TV+W','ZF'}); 
grid on

%% Graph of HFEN
HFENzf = HFEN(zf)*ones(counter,1);
figure(11); clf;
h=plot(1:1:counter,HFENvec_TGVW ,'k'); 
set(h,'LineWidth',2);
hold on;
h=plot(1:1:counter,HFENvec_TGV ,'.-'); 
set(h,'LineWidth',2);
hold on;
h=plot(1:1:counter,HFENvec_TVW ,':'); 
set(h,'LineWidth',2);
hold on;
h=plot(1:1:counter,HFENzf,'--'); 
set(h,'LineWidth',2);
set(gca,'FontSize',14);
h=xlabel('Iterations');
set(h,'FontSize',14);
h=ylabel('HFEN');
set(h,'FontSize',14);
legend({'Proposed','TGV','TV+W','ZF'}); 
grid on

%% Graph of SSIM
SSIMzf = ssim(abs(zf),abs(x0))*ones(counter,1);
figure(12); clf;
h=plot(1:1:counter,SSIM_TGVW,'k'); 
set(h,'LineWidth',2);
hold on;
h=plot(1:1:counter,SSIM_TGV,'-.'); 
set(h,'LineWidth',2);
hold on;
h=plot(1:1:counter,SSIM_TVW,':'); 
set(h,'LineWidth',2);
hold on;
h=plot(1:1:counter,SSIMzf,'--'); 
set(h,'LineWidth',2);
set(gca,'FontSize',14);
h=xlabel('Iterations');
set(h,'FontSize',14);
h=ylabel('SSIM');
set(h,'FontSize',14);
legend({'Proposed','TGV','TV+W','ZF'}); 
grid on

%% Graph of Primal Energy
figure(13); clf;
h=plot(1:1:counter,PE_TGVW,'k'); 
set(h,'LineWidth',2);
hold on;
set(gca,'FontSize',14);
h=xlabel('Iteration');
set(h,'FontSize',14);
h=ylabel('Energy of the Objective Function');
set(h,'FontSize',14);
legend({'W+TGV'}); 
grid on

%% Outputs
out.IterationsCount = counter;
out.SamplingRate = numel(find(mask))/numel(mask);

out.TVW_Runtime = time_TVW;
out.TGV_Runtime = time_TGV;
out.TGVW_Runtime = time_TGVW;

out.SNR_zf = SNRzf(end);
out.TVW_SNR = SNR_TVW(end);
out.TGV_SNR = SNR_TGV(end);
out.TGVW_SNR = SNR_TGVW(end);

out.SSIM_zf = SSIMzf(end);
out.TVW_SSIM = SSIM_TVW(end);
out.TGV_SSIM = SSIM_TGV(end);
out.TGVW_SSIM = SSIM_TGVW(end);

out.HFEN_zf = HFENzf(end);
out.TVW_HFEN = HFENvec_TVW(end);
out.TGV_HFEN = HFENvec_TGV(end);
out.TGVW_HFEN = HFENvec_TGVW(end);

function z = E(p)
global m n
z = zeros(m,n,3);
z(:,:,1) = dxm(p(:,:,1));
z(:,:,2) = dym(p(:,:,2));
z(:,:,3) = (dym(p(:,:,1)) + dxm(p(:,:,2)))/2;

function r = div_2(z)
global m n
r = zeros(m,n,2);
r(:,:,1) = dxp(z(:,:,1)) + dyp(z(:,:,3));
r(:,:,2) = dxp(z(:,:,3)) + dyp(z(:,:,2));

function u = projC(u,c)
absu = norm(u,'fro');
denom = max(1,absu/c);
u = u./denom;

% function x = projC2(x,bb)
% a = x(:);
% Pbox = min(max(a,0),255);
% aTPbox = a'*Pbox;
% if aTPbox <= bb
%     x = Pbox;
% else
%     lstar = ;
%     x = min(max(a*(1-mustar),0),255);
% end

function p = projP(p,alpha1)

  absp = sqrt(abs(p(:,:,1)).^2 + abs(p(:,:,2)).^2);
  denom = max(1,absp/alpha1);
  p(:,:,1) = p(:,:,1)./denom;
  p(:,:,2) = p(:,:,2)./denom;  

function q = projQ(q,alpha0)
  absq = sqrt(abs(q(:,:,1)).^2 + abs(q(:,:,2)).^2 + 2*abs(q(:,:,3)).^2);
  denom = max(1,absq/alpha0);
  q(:,:,1) = q(:,:,1)./denom;
  q(:,:,2) = q(:,:,2)./denom;
  q(:,:,3) = q(:,:,3)./denom;  
 
 function pe = objTGVW(u,v)
 global alpha0 alpha1 lambda beta b D R W;
 p = D(u)-v;
 q = E(v);
 pe = (1/2*lambda)*norm(R(u)-b,'fro')^2 + alpha1*sum(sum(sqrt(abs(p(:,:,1)).^2 + abs(p(:,:,2)).^2)))...
      + alpha0*sum(sum(sqrt(abs(q(:,:,1)).^2 + abs(q(:,:,2)).^2 + 2*abs(q(:,:,3)).^2)))...
      + beta*sum(sum(abs(W*u)));
              
 function pe = objTGV(u,v)
 global alpha0 alpha1 lambda b D R;
 p = D(u)-v;
 q = E(v);
 pe = (1/2*lambda)*norm(R(u)-b,'fro')^2 + alpha1*sum(sum(sqrt(abs(p(:,:,1)).^2 + abs(p(:,:,2)).^2)))...
      + alpha0*sum(sum(sqrt(abs(q(:,:,1)).^2 + abs(q(:,:,2)).^2 + 2*abs(q(:,:,3)).^2)));

 function pe = objTVW(x)
 global alpha betaTVW R b W D 
        pe =  norm(R(x)-b,'fro')^2/2+alpha*sum(sum(sqrt(sum(D(x).^2,3))))...
            + betaTVW*sum(sum(abs(W*x)));