function full_extra21_worker(worker, workers)
% One disjoint shard of the 21 supplemental shallow image features.
here=fileparts(mfilename('fullpath')); jobs=jsondecode(fileread(fullfile(here,'jobs.json'))); done=0;
for k=(worker+1):workers:numel(jobs)
    job=jobs(k); target=fullfile(here,'per_image',[job.key '.json']); if isfile(target), continue; end
    timer=tic; out=struct('key',job.key,'sha256',job.sha256,'status','ok','values',[],'error','','elapsed_seconds',0);
    try
        [rgb,map,alpha]=imread(job.path);
        if ~isempty(map)||~isempty(alpha), error('extra21:UnsupportedImage','Indexed/alpha image requires explicit policy'); end
        if ismatrix(rgb), rgb=repmat(rgb,1,1,3); end
        if ~isa(rgb,'uint8')||size(rgb,3)~=3, error('extra21:UnsupportedImage','Expected uint8 RGB/grayscale'); end
        values=extra_shallow21(rgb);
        if numel(values)~=21||any(~isfinite(values)), error('extra21:InvalidFeatures','Wrong dimension or non-finite feature value'); end
        out.values=values;
    catch err, out.status='failed'; out.error=[err.identifier ': ' err.message]; end
    out.elapsed_seconds=toc(timer); tmp=[target '.tmp']; fid=fopen(tmp,'w','n','UTF-8'); if fid<0,error('Cannot write checkpoint');end; fprintf(fid,'%s',jsonencode(out)); fclose(fid); movefile(tmp,target);
    done=done+1; if mod(done,25)==0, fprintf('worker=%d completed=%d image=%d/%d elapsed=%.3fs status=%s\n',worker,done,k,numel(jobs),out.elapsed_seconds,out.status);end
end
fprintf('worker=%d finished newly_completed=%d\n',worker,done);
end

function values=extra_shallow21(rgb)
im=im2double(rgb); hsv=rgb2hsv(im); h=hsv(:,:,1); s=hsv(:,:,2); v=hsv(:,:,3);
rg=im(:,:,1)-im(:,:,2); yb=.5*(im(:,:,1)+im(:,:,2))-im(:,:,3);
colourful=sqrt(std(rg(:))^2+std(yb(:))^2)+.3*sqrt(mean(rg(:))^2+mean(yb(:))^2);
black=v<.15; white=~black&s<.15&v>=.85; gray=~black&~white&s<.15; c=~black&~white&~gray&s>=.15;
brown=c&h>=.04&h<.14&v<.65; orange=c&h>=.04&h<.14&~brown; yellow=c&h>=.14&h<.19; green=c&h>=.19&h<.47;
blue=c&h>=.47&h<.72; purple=c&h>=.72&h<.87; pink=c&h>=.87&h<.96; red=c&(h>=.96|h<.04);
props=[mean(black,'all'),mean(blue,'all'),mean(brown,'all'),mean(gray,'all'),mean(green,'all'),mean(orange,'all'),mean(pink,'all'),mean(purple,'all'),mean(red,'all'),mean(white,'all'),mean(yellow,'all')];
grayim=rgb2gray(rgb); edge_fraction=mean(edge(grayim,'Canny'),'all'); local_entropy=mean(entropyfilt(grayim),'all');
[labels,~]=superpixels(rgb,250,'Compactness',10); actual_regions=numel(unique(labels)); region_size_mean=numel(labels)/actual_regions; tamura=tamura3sigs_no_stats(rgb);
values=[colourful,props,edge_fraction,actual_regions,region_size_mean,numel(grayim),size(rgb,2)/size(rgb,1),local_entropy,tamura(:)'];
end
