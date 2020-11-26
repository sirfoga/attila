cwd				:= $(shell pwd)


clean::
	rm out/

cleanh5::
	for f in $$(find . -name "*.h5"); do rm $${f}; done

r-update::
	git reset --hard origin/main && git pull

get-out::
	scp -r stfo194b@taurus.hrsk.tu-dresden.de:/home/stfo194b/martin/attila/out .
