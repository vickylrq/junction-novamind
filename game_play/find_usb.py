
import usb.core
import usb.util

dev = usb.core.find(idVendor=0x1058, idProduct=0x264F)
if dev is None:
    raise ValueError('Device not found')
print(dev)
# set the active configuration. With no arguments, the first
# configuration will be the active one
dev.set_configuration()

# get an endpoint instance
cfg = dev.get_active_configuration()
intf = cfg[(0,0)]

ep = usb.util.find_descriptor(
    intf,
    # match the first OUT endpoint
    custom_match = \
    lambda e: \
        usb.util.endpoint_direction(e.bEndpointAddress) == \
        usb.util.ENDPOINT_OUT)
print(ep)
assert ep is not None

# write the data
ep.write('test')
