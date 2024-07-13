close all; clearvars; clc;
fid = fopen('./result/output_correction_resnet50.csv','r');
tmp = textscan(fid,'%s%f%f%f%f%f%f%f%f','delimiter',',','headerlines',1);
fclose(fid);
filename = tmp{1};
yt = tmp{2}; yp = tmp{3};
score = [tmp{4},tmp{5},tmp{6},tmp{7},tmp{8},tmp{9}];
score = exp(score);
for ii = 1:size(score,1)
    score(ii,:) = score(ii,:)/sum(score(ii,:));
end

for kk = 1:size(score,1)
    fprintf(1,'%s ',filename{kk});
    fprintf(1,'%5.3f, ', 100*score(kk,:));
    fprintf(1,'\n')
end
