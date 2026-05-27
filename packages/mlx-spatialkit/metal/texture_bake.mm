#include "metal_probe.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace mlx_spatialkit {

bool metal_device_available() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device != nil;
  }
}

std::string metal_device_name() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      return "unavailable";
    }
    NSString *name = [device name];
    if (name == nil) {
      return "unknown";
    }
    return std::string([name UTF8String]);
  }
}

}  // namespace mlx_spatialkit
