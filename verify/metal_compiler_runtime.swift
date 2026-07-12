import Foundation
import Metal

func fail(_ message: String) -> Never { fputs(message + "\n", stderr); exit(1) }
guard let device = MTLCreateSystemDefaultDevice() else { fail("No Metal device") }
guard let queue = device.makeCommandQueue() else { fail("No Metal command queue") }
let env = ProcessInfo.processInfo.environment
guard let ewiseSource = env["NUM_MSL_EWISE"], let reduceSource = env["NUM_MSL_REDUCE"] else {
  fail("Compiler MSL sources were not provided")
}
func pipeline(_ source: String, _ name: String) -> MTLComputePipelineState {
  do {
    let library = try device.makeLibrary(source: source, options: nil)
    guard let function = library.makeFunction(name: name) else { fail("Missing function " + name) }
    return try device.makeComputePipelineState(function: function)
  } catch { fail("Metal compile/pipeline error: \(error)") }
}
func buffer(_ values: [Float]) -> MTLBuffer {
  let copy = values
  return copy.withUnsafeBytes { bytes in
    guard let result = device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count,
                                         options: .storageModeShared) else { fail("Buffer allocation") }
    return result
  }
}
let add = pipeline(ewiseSource, "num_ewise_add_f32")
let sum = pipeline(reduceSource, "num_reduce_sum_f32")
let x = buffer([1,2,3,4]), y = buffer([10,20,30,40])
guard let z = device.makeBuffer(length: 4 * MemoryLayout<Float>.size, options: .storageModeShared),
      let parts = device.makeBuffer(length: MemoryLayout<Float>.size, options: .storageModeShared),
      let nbuf = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModeShared) else { fail("Output allocation") }
nbuf.contents().assumingMemoryBound(to: UInt32.self).pointee = 4
guard let commands = queue.makeCommandBuffer() else { fail("Command buffer") }
guard let addEncoder = commands.makeComputeCommandEncoder() else { fail("Add encoder") }
addEncoder.setComputePipelineState(add)
addEncoder.setBuffer(x, offset: 0, index: 0); addEncoder.setBuffer(y, offset: 0, index: 1)
addEncoder.setBuffer(z, offset: 0, index: 2); addEncoder.setBuffer(nbuf, offset: 0, index: 3)
addEncoder.dispatchThreads(MTLSize(width: 4, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 4, height: 1, depth: 1))
addEncoder.endEncoding()
guard let reduceEncoder = commands.makeComputeCommandEncoder() else { fail("Reduce encoder") }
reduceEncoder.setComputePipelineState(sum)
reduceEncoder.setBuffer(x, offset: 0, index: 0); reduceEncoder.setBuffer(parts, offset: 0, index: 1)
reduceEncoder.setBuffer(nbuf, offset: 0, index: 2); reduceEncoder.setThreadgroupMemoryLength(256 * MemoryLayout<Float>.size, index: 0)
reduceEncoder.dispatchThreads(MTLSize(width: 256, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
reduceEncoder.endEncoding(); commands.commit(); commands.waitUntilCompleted()
if let error = commands.error { fail("Metal command failure: \(error)") }
let zv = z.contents().bindMemory(to: Float.self, capacity: 4)
let actual = (0..<4).map { zv[$0] }, expected: [Float] = [11,22,33,44]
guard zip(actual, expected).allSatisfy({ abs($0 - $1) < 1e-5 }) else { fail("Ewise mismatch \(actual)") }
let reduced = parts.contents().assumingMemoryBound(to: Float.self).pointee
guard abs(reduced - 10) < 1e-5 else { fail("Reduction mismatch \(reduced)") }
print("Metal compiler kernels: PASS device=\(device.name) ewise=\(actual) sum=\(reduced)")
