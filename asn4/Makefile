all: ref.elf

VBXSIM=mxp/repository/lib/vbxsim
VBXAPI=mxp/repository/lib/vbxapi

ref.elf: ref.c $(VBXAPI)/libvbxapi.a $(VBXSIM)/libvbxsim.a
	gcc -Wall -g -std=c99 -DVBX_SIMULATOR -I$(VBXAPI) -o $@ $^

$(VBXAPI)/libvbxapi.a  $(VBXSIM)/libvbxsim.a:
	make -C $(dir $@) SIMULATOR=true

clean:
	rm -rf ref.elf
	make -C $(VBXAPI) clean SIMULATOR=true
	make -C $(VBXSIM) clean SIMULATOR=true

.phony: all clean
