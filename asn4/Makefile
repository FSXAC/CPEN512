all: mm.elf

VBXSIM=mxp/repository/lib/vbxsim
VBXAPI=mxp/repository/lib/vbxapi

mm.elf: mm.c $(VBXAPI)/libvbxapi.a $(VBXSIM)/libvbxsim.a
	gcc -Wall -g -O0 -std=c99 -DVBX_SIMULATOR -I$(VBXAPI) -o $@ $^

$(VBXAPI)/libvbxapi.a  $(VBXSIM)/libvbxsim.a:
	make -C $(dir $@) SIMULATOR=true

clean:
	rm -rf ref.elf
	make -C $(VBXAPI) clean SIMULATOR=true
	make -C $(VBXSIM) clean SIMULATOR=true

.phony: all clean
