cwd				:= $(shell pwd)
out_folder		:= ./out

clean::
	rm -rf ${out_folder}

cleanH5::
	for f in $$(find ${out_folder} -name "*.h5"); do rm $${f}; done

cleanJson::
	for f in $$(find ${out_folder} -name "*.json"); do rm $${f}; done

cleanTex::
	for f in $$(find ${out_folder} -name "*.tex"); do rm $${f}; done

r-update::
	git reset --hard origin/main && git pull

get-out::
	scp -r stfo194b@taurus.hrsk.tu-dresden.de:/home/stfo194b/martin/attila/out .

srun::
	srun \
	--partition=ml \
	--nodes=1 \
	--tasks=1 \
	--cpus-per-task=2 \
	--gres=gpu:1 \
	--mem-per-cpu=2583 \
	--time=01:00:00 \
	--account=p_ml_cv \
	--pty bash

hello::
	@echo "hello ${USER} !"
