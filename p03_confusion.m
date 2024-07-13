function p03_confusion()
    filename = './result/output_fj_resnet50.cm6';
    [cm6,cm6_] = load_confusion(filename);
    plot_confusion(cm6,cm6_);
end


function [cm,cm_normal] = load_confusion(file)
    cm = load(file);
    [m,n] = size(cm);
    cm2 = zeros(6,6);
    cm2(1:m,1:n) = cm; cm = cm2;
    cm_normal = cm*0;
    for i = 1:m
        cm_normal(i,:) = cm(i,:)/( sum(cm(i,:))+1e-11 );
    end
end

function plot_confusion(cm6,cm6_)
    figure(1); confusionchart(cm6);
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
    figure(2); pa = patch('vertices',v,'faces',e,'cdata',tmp(:),'edgecolor','k','linewidth',1,...
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
end