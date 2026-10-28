/// Preallocated float32 circular buffer (port of src/soundwatch/audio/ringbuffer.py).
struct RingBuffer {
    let capacity: Int
    private var buf: [Float]
    private var pos = 0  // next write index == oldest sample once full
    private(set) var totalWritten = 0

    init(capacity: Int) {
        precondition(capacity > 0)
        self.capacity = capacity
        buf = [Float](repeating: 0, count: capacity)
    }

    mutating func clear() {
        for i in 0..<capacity { buf[i] = 0 }
        pos = 0
        totalWritten = 0
    }

    mutating func write(_ x: UnsafeBufferPointer<Float>) {
        let n = x.count
        guard n > 0 else { return }
        if n >= capacity {
            for i in 0..<capacity { buf[i] = x[n - capacity + i] }
            pos = 0
        } else {
            let first = min(n, capacity - pos)
            for i in 0..<first { buf[pos + i] = x[i] }
            for i in first..<n { buf[i - first] = x[i] }
            pos = (pos + n) % capacity
        }
        totalWritten += n
    }

    mutating func write(_ x: [Float]) {
        x.withUnsafeBufferPointer { write($0) }
    }

    /// Most recent `capacity` samples in chronological order (zeros first if not yet full).
    func latest(into out: inout [Float]) {
        if out.count != capacity { out = [Float](repeating: 0, count: capacity) }
        let tail = capacity - pos
        for i in 0..<tail { out[i] = buf[pos + i] }
        for i in 0..<pos { out[tail + i] = buf[i] }
    }
}
