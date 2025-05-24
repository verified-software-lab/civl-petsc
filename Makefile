# This makefile looks for all the subdirectories inside the "targets" folder
# (it skips hidden folders and the folder called "CIVLREP").
SUBDIRS := $(shell find targets -mindepth 1 -maxdepth 1 -type d -not -name ".*" -not -name "CIVLREP" -exec basename {} \;)

# The "small" target runs tests in each subfolder.
# It runs the verification tests with vector sizes from 1..3 and processor counts from 1 and 2,
# covering both real and complex cases.
small:
	@count=0; for dir in $(SUBDIRS); do \
		count=$$((count+1)); \
		echo "\n=============================================="; \
		printf "=== Verifying: %s (%d/%d)\n" "$$dir" "$$count" "$(words $(SUBDIRS))"; \
		echo "==============================================\n"; \
		$(MAKE) -C targets/$$dir small; \
	done

# The "big" target runs tests in each subfolder.
# It runs the verification tests with vector sizes from 1..5 and processor counts from 1..5,
# covering both real and complex cases. For some function it nprocs and vecsize are reduced to 1..4
# to avoid long verification times.
big:
	@count=0; for dir in $(SUBDIRS); do \
		count=$$((count+1)); \
		echo "\n=============================================="; \
		printf "=== Verifying: %s (%d/%d)\n" "$$dir" "$$count" "$(words $(SUBDIRS))"; \
		echo "==============================================\n"; \
		$(MAKE) -C targets/$$dir big; \
	done

# The "clean" target goes into every subfolder and runs its clean target
clean:
	@for d in $(SUBDIRS); do \
		$(MAKE) -C targets/$$d clean; \
	done; \

.phony: small big clean
