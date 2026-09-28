function full_osf37_worker(worker, workers)
% Original OSF formula calls; each MATLAB process owns a disjoint image shard.
here=fileparts(mfilename('fullpath'));
cfg=jsondecode(fileread(fullfile(here,'config.json')));
jobs=jsondecode(fileread(fullfile(here,'jobs.json')));
addpath(cfg.reference,'-begin'); cleanup=onCleanup(@() rmpath(cfg.reference));
for f={'use_of_light','indicators','pda','huestats','trt','low_depth_of_field_indicators','wavelet_features','greylev'}
    if ~strcmp(which(f{1}),fullfile(cfg.reference,[f{1} '.m'])); error('Unexpected function resolution: %s',f{1}); end
end
dwtmode('sym','nodisp'); done=0;
for k=(worker+1):workers:numel(jobs)
    job=jobs(k); target=fullfile(here,'per_image',[job.key '.json']); if isfile(target), continue; end
    t=tic; out=struct('key',job.key,'sha256',job.sha256,'status','ok','values',[],'error','','elapsed_seconds',0);
    try
        [rgb,map,alpha]=imread(job.path);
        if ~isempty(map)||~isempty(alpha), error('osf37:UnsupportedImage','Indexed/alpha image requires explicit policy'); end
        if ismatrix(rgb), rgb=repmat(rgb,1,1,3); end
        if ~isa(rgb,'uint8')||size(rgb,3)~=3, error('osf37:UnsupportedImage','Expected uint8 RGB/grayscale'); end
        stats=indicators(rgb); [p,d,a]=pda(rgb); dof=low_depth_of_field_indicators(rgb); thirds=trt(rgb); wave=wavelet_features(rgb); glcm=greylev(rgb);
        values=[use_of_light(rgb),stats(:)',p,d,a,huestats(rgb),sum(dof(1:3)),sum(dof(4:6)),sum(dof(7:9)),thirds(2),thirds(3),wave(:)',glcm(:)'];
        if numel(values)~=37||any(~isfinite(values)), error('osf37:InvalidFeatures','Wrong dimension or non-finite output'); end
        out.values=values;
    catch err, out.status='failed'; out.error=[err.identifier ': ' err.message]; end
    out.elapsed_seconds=toc(t); temp=[target '.tmp']; fid=fopen(temp,'w','n','UTF-8'); if fid<0,error('Cannot open output');end; fprintf(fid,'%s',jsonencode(out)); fclose(fid); movefile(temp,target);
    done=done+1; if mod(done,25)==0, fprintf('worker=%d new_completed=%d image=%d/%d\n',worker,done,k,numel(jobs));end
end
fprintf('worker=%d finished new_completed=%d\n',worker,done);
end
