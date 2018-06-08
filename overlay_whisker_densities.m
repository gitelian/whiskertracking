
rows = 2;
cols = 9;
k = 1;


for j = 1:cols
    temp = zeros(size(whisker_density(:,:,j),1), size(whisker_density(:,:,j+cols),2),3);
    a = imadjust(1-whisker_density(:,:,j),[0.95 1],[]);
    b = imadjust(1-whisker_density(:,:,j+9),[0.95 1],[]);
    h = figure;
    imshowpair(a, b, 'falsecolor')
    title(['Pos ' num2str(j)])
    set(h,'PaperPositionMode','auto')
    saveas(h, ['~/Documents/AdesnikLab/Figures/HSV_Reflective/overlay' sprintf('%02d',j) '.tif'])
    close(h)
end
