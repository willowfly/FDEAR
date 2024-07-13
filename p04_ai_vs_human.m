close all; clearvars; clc;
res = [
    0.792	0.797	0.958	0.957
    0.700	0.706	0.942	0.938
    0.792	0.788	0.983	0.983
    0.758	0.757	0.867	0.875
    0.642	0.639	0.858	0.866
    0.450	0.408	0.875	0.855
    0.642	0.628	0.892	0.874
    0.617	0.619	0.667	0.708
    0.450	0.444	0.842	0.789
    0.542	0.531	0.858	0.846
    0.533	0.522	0.750	0.779
    0.458	0.442	0.842	0.800
    0.833	0.832	0.942	0.943
    0.842	0.836	0.958	0.959
    0.850	0.846	0.975	0.976
    0.842	0.838	0.967	0.967    
]; 

human1 = res(1:4,:);
human2 = res(5:8,:);
human3 = res(9:12,:);
human4 = res(13:16,:);

figure(1); set(gcf,'position',[0,0,600,800]); 
mm = 1; 
tmp = [human1(:,mm),human2(:,mm),human3(:,mm),human4(:,mm)];
boxplot(tmp,'notch','off'); hold on;
plot(mean(tmp),'kx');
for i = 1:4
    plot( i+0.1*[-1.5,-0.5,0.5,1.5], res(4*(i-1)+[1:4],mm ), 'ko', 'markersize',5);
end
set(gca,'ylim',[0.3,1.0]);

figure(2); set(gcf,'position',[0,0,600,800]);
mm = 3; 
tmp = [human1(:,mm),human2(:,mm),human3(:,mm),human4(:,mm)];
boxplot(tmp,'notch','off'); hold on;
plot(mean(tmp),'kx');
for i = 1:4
    plot( i+0.1*[-1.5,-0.5,0.5,1.5], res(4*(i-1)+[1:4],mm ), 'ko', 'markersize',5);
end
set(gca,'ylim',[0.6,1.0]);

%% confusion matrix
fid = fopen('data120_human_res/res_rename/true_result.csv','r');
tmp = textscan(fid,'%d%d%s','delimiter',',','headerlines',1);
fclose(fid);
yt = tmp{2};
YP = [];
for i = 1:4
    filename = sprintf('data120_human_res/res_rename/diag_profession_%02d.csv',i);
    fid = fopen(filename,'r');
    tmp = textscan(fid,'%d%s','delimiter',':'); fclose(fid);
    yp_label = tmp{2}; yp_label = yp_label(2:121);
    yp = zeros(120,1); 
    for j = 1:120
        yp(j) = find(yp_label{j}=='abdcefN');
    end
    YP = [YP;yp];
end
YT = cast(repmat(yt,4,1),'double');
cm6 = confusionmat(YT,YP);
cm6_ = cm6*0;
for ii = 1:6
    cm6_(ii,:) = cm6(ii,:)/sum(cm6(ii,:));
end


figure(3); confusionchart(cm6);
v = zeros(7*7,2); k = 0;
for j = 1:7
    for i = 1:7
        v(k+1,:) = [i,j];
        k = k+1;
    end
end
e = zeros(6*6,4); k = 0;
for j = 1:6
    for i = 1:6
        n1 = 7*(j-1)+i; n2 = n1+1; n3=n2+7; n4=n3-1;
        e(k+1,:) = [n1,n2,n3,n4]; k = k+1;
    end
end
tmp = cm6_';
figure(4); pa = patch('vertices',v,'faces',e,'cdata',tmp(:),'edgecolor','k','linewidth',1,...
    'facecolor','flat'); hold on;
axis equal;
set(gca,'xlim',[1,7],'ylim',[1,7],'ydir','reverse'); axis off;
crp = hot(100); crp = flipud(crp);
colormap(crp); clim([0,1]);

for i = 1:6
    for j = 1:6
        txt = sprintf('%3.1f%%\n%d',100*cm6_(j,i),cm6(j,i));
        text(i+0.5,j+0.5,txt,'fontname','arial narrow','fontsize',11,'HorizontalAlignment','center','VerticalAlignment','middle');
    end
end
