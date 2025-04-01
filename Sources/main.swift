//
//  main.swift
//  llama2
//
//  Created by Angel Genov on 2023-08-18.
//
// swift run -c release llama2 stories110M.bin


import Foundation
import ArgumentParser

extension Array where Element: Copyable {
    mutating func copyFromSlice(slice: ArraySlice<Element>) {
        guard slice.count == self.count else {print ("Slice count does not match array count"); return }
        for (i, e) in slice.enumerated() {
            self[i] = e
        }
    }
}

extension FileHandle {
    func readFloatArrayWithSize(_ size: Int) -> [Float]? {
        guard let data = try? self.read(upToCount: size * MemoryLayout<Float>.size) else {
            print("Could not read config file")
            return [Float]()
        }
        
        let buffer = data.withUnsafeBytes{
            (configData: UnsafeRawBufferPointer) -> [Float] in
            configData.bindMemory(to: Float.self).map {Float($0)}
        }
        
        return buffer
    }

    func readFloat() -> Float {
        guard let data = try? self.read(upToCount: MemoryLayout<Float>.size) else {
            print("Could not read config file")
            return 0.0
        }
        
        let buffer = data.withUnsafeBytes{
            (configData: UnsafeRawBufferPointer) -> Float in
            configData.bindMemory(to: Float.self).first!
        }
        
        return buffer
    }

    func readInt() -> Int {
        guard let data = try? self.read(upToCount: MemoryLayout<CInt>.size) else {
            print("Could not read config file")
            return 0
        }
        
        let buffer = data.withUnsafeBytes{
            (configData: UnsafeRawBufferPointer) -> CInt in
            configData.bindMemory(to: CInt.self).first!
        }
        
        return Int(buffer)
    }

    func readUit8ArrayWithSize(_ size: Int) -> [UInt8] {
        guard let data = try? self.read(upToCount: size * MemoryLayout<UInt8>.size) else {
            print("Could not read config file")
            return [UInt8]()
        }
        
        let buffer = data.withUnsafeBytes{
            (configData: UnsafeRawBufferPointer) -> [UInt8] in
            configData.bindMemory(to: UInt8.self).map {UInt8($0)}
        }
        
        return buffer
    }
}

// MARK: -  Transformer and RunState structs
struct Config {
    let SIZE = 7
    
    let dim: Int // transformer dimension
    let hidden_dim: Int // for ffn layers
    let n_layers: Int // number of layers
    let n_heads: Int // number of query heads
    let n_kv_heads: Int // number of key/value heads (can be < query heads because of multiquery)
    let vocab_size: Int // vocabulary size, usually 256 (byte-level)
    let seq_len: Int // max sequence length
    
    init?(with file: FileHandle) {
        guard let configData = try? file.read(upToCount: SIZE * MemoryLayout<CInt>.size) else {
            return nil
        }
        
        let buffer = configData.withUnsafeBytes{
            (configData: UnsafeRawBufferPointer) -> [CInt] in
            configData.bindMemory(to: CInt.self).map {CInt($0)}
        }
        
        dim = Int(buffer[0])
        hidden_dim = Int(buffer[1])
        n_layers = Int(buffer[2])
        n_heads = Int(buffer[3])
        n_kv_heads = Int(buffer[4])
        vocab_size = Int(buffer[5])
        seq_len = Int(buffer[6])
    }
}

struct TransformerWeights {
    // token embedding table
    let token_embedding_table: [Float]   // (vocab_size, dim)
    // weights for rmsnorms
    let rms_att_weight: [Float] // (layer, dim) rmsnorm weights
    let rms_ffn_weight: [Float] // (layer, dim)
    // weights for matmuls
    let wq: [Float] // (layer, dim, dim)
    let wk: [Float] // (layer, dim, dim)
    let wv: [Float] // (layer, dim, dim)
    let wo: [Float] // (layer, dim, dim)
    // weights for ffn
    let w1: [Float] // (layer, hidden_dim, dim)
    let w2: [Float] // (layer, dim, hidden_dim)
    let w3: [Float] // (layer, hidden_dim, dim)
    // final rmsnorm
    let rms_final_weight: [Float] // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    let freq_cis_real: [Float] // (seq_len, head_size/2)
    let freq_cis_imag: [Float] // (seq_len, head_size/2)
    // (optional) classifier weights for the logits, on the last layer
    let wcls: [Float]
    
    init?(with file: FileHandle, config: Config) {
        var size: Int = 0
        
        // (1) token embedding table
        size = config.dim * config.vocab_size
        guard let token_embedding_table = file.readFloatArrayWithSize(size) else {
            return nil
        }
        self.token_embedding_table = token_embedding_table
        
        // weights for rmsnorms
        size = config.n_layers * config.dim
        guard let rms_att_weight = file.readFloatArrayWithSize(size) else {
            return nil
        }
        self.rms_att_weight = rms_att_weight
        
        // (2) weights for matmuls
        size = config.n_layers * config.dim * config.dim
        guard let wq = file.readFloatArrayWithSize(size) else {
            return nil
        }
        self.wq = wq
        
        size = config.n_layers * config.dim * config.dim
        guard let wk = file.readFloatArrayWithSize(size) else {
            return nil
        }
        self.wk = wk
        
        size = config.n_layers * config.dim * config.dim
        guard let wv = file.readFloatArrayWithSize(size) else {
            return nil
        }
        self.wv = wv
        
        size = config.n_layers * config.dim * config.dim
        guard let wo = file.readFloatArrayWithSize(size) else {
            return nil
        }
        self.wo = wo
        
        // (3) weigh.ts for rms_ffn_weight
        size = config.n_layers * config.dim
        guard let rms_ffn_weight = file.readFloatArrayWithSize(size) else {
            return nil
        }
        self.rms_ffn_weight = rms_ffn_weight
        
        // (4) weights for ffn
        size = config.n_layers * config.hidden_dim * config.dim
        guard let w1 = file.readFloatArrayWithSize(size) else {
            return nil
        }
        self.w1 = w1
        
        size = config.n_layers * config.hidden_dim * config.dim
        guard let w2 = file.readFloatArrayWithSize(size) else {
            return nil
        }
        self.w2 = w2
        
        size = config.n_layers * config.hidden_dim * config.dim
        guard let w3 = file.readFloatArrayWithSize(size) else {
            return nil
        }
        self.w3 = w3
        
        // (5) final rmsnorm
        size = config.dim
        guard let rms_final_weight = file.readFloatArrayWithSize(size) else {
            return nil
        }
        self.rms_final_weight = rms_final_weight
        
        // (6) freq_cis for RoPE relatively positional embeddings
        let headSize: Int = config.dim / config.n_heads
        size = config.seq_len * headSize / 2
        guard let freq_cis_real = file.readFloatArrayWithSize(size) else {
            return nil
        }
        self.freq_cis_real = freq_cis_real
        
        size = config.seq_len * headSize / 2
        guard let freq_cis_imag = file.readFloatArrayWithSize(size) else {
            return nil
        }
        self.freq_cis_imag = freq_cis_imag
        
        // (optional) classifier weights for the logits, on the last layer
        wcls = token_embedding_table
    }
}

/// struct used when sorting probabilities during top-p sampling
struct ProbIndex {
    var prob: Float = 0.0
    var index: Int = 0
}

/// current wave of activations
struct RunState {
    var x: [Float] // activation at current time stamp (dim,)
    var xb: [Float] // same, but inside a residual branch (dim,)
    var xb2: [Float] // an additional buffer just for convenience (dim,)
    var hb: [Float] // buffer for hidden dimension in the ffn (hidden_dim,)
    var hb2: [Float] // buffer for hidden dimension in the ffn (hidden_dim,)
    var q: [Float] // query (dim,)
    var k: [Float] // key (dim,)
    var v: [Float] // value (dim,)
    var att: [Float] // buffer for scores/attention values (n_heads, seq_len)
    var logits: [Float] // output logits
    var probindex: [ProbIndex] // buffer used in top-p sampling
    // kv cache
    var keyCache: [Float] // (layer, seq_len, dim)
    var valueCache: [Float] // (layer, seq_len, dim)
    
    init(config: Config) {
        x = Array(repeating: 0.0, count: config.dim)
        xb = Array(repeating: 0.0, count: config.dim)
        xb2 = Array(repeating: 0.0, count: config.dim)
        hb = Array(repeating: 0.0, count: config.hidden_dim)
        hb2 = Array(repeating: 0.0, count: config.hidden_dim)
        q = Array(repeating: 0.0, count: config.dim)
        k = Array(repeating: 0.0, count: config.dim)
        v = Array(repeating: 0.0, count: config.dim)
        att = Array(repeating: 0.0, count: config.n_heads * config.seq_len)
        logits = Array(repeating: 0.0, count: config.vocab_size)
        probindex = Array(repeating: ProbIndex(), count: config.vocab_size)
        keyCache = Array(repeating: 0.0, count: config.n_layers * config.seq_len * config.dim)
        valueCache = Array(repeating: 0.0, count: config.n_layers * config.seq_len * config.dim)
    }
}

/// byte pair encoding (BPE) tokenizer, encodes strings into tokens so we can prompt
struct BPETokenizer {
    var vocab: [String] = []
    var vocabScores: [Float] = []
    var tokenIndex: [String: Int] = [:]
    var maxTokenLen: Int = 0
    
    mutating func loadVocab(fileName: String, vocabSize: Int) {
        vocab = []
        vocabScores = []
        tokenIndex = [:]
        
        guard let url = URL(string: fileName) else {
            print("Invalid URL: \(fileName)")
            exit(2)
        }
        
        guard let file = try? FileHandle(forReadingFrom: url) else {
            print("Could not read vocab file: \(fileName)")
            exit(2)
        }
        
        maxTokenLen = file.readInt()
        
        for i in 0..<vocabSize {
            let score = file.readFloat()
            let len = file.readInt()
            let buff = file.readUit8ArrayWithSize(len)
            let token = String(decoding: buff, as: Unicode.UTF8.self)
            vocab.append(token)
            vocabScores.append(score)
            if (tokenIndex[token] == nil) {
                // get the first maching
                tokenIndex[token] = i
            }
        }
    }
    
    func bpeEncode(_ input: String) -> [Int] {
        var result: [Int] = []
        
        for c in input {
            guard let t = tokenIndex[String(c)] else {
                print("Invalid character: \(c)")
                exit(3)
            }
            result.append(t)
        }
        
        while (true) {
            var bestScore = Float(-1e10)
            var bestId = Int(-1)
            var bestIdx = Int(-1)
            
            for i in 0..<result.count-1 {
                let tmp = vocab[result[i]] + vocab[result[i+1]]
                if let id = tokenIndex[tmp] {
                    if (vocabScores[id] > bestScore) {
                        bestScore = vocabScores[id]
                        bestId = id
                        bestIdx = i
                    }
                }
            }
            
            if (bestIdx == -1) {
                break
            }
            
            result[bestIdx] = bestId
            result.remove(at: bestIdx+1)
        }
        
        return result
    }
}

// MARK: - Neural Net Blocks
func accum(out: inout Array<Float>, b: Array<Float>) {
    assert(out.count == b.count)
    for i in 0..<out.count {
        out[i] = out[i] + b[i]
    }
}

func rmsnorn(out: inout Array<Float>, x: Array<Float>, weights: ArraySlice<Float>) {
    assert((out.count == x.count) && (x.count == weights.count))
    var ss:Float = 0.0
    // calculate sum of squares
    for f in x {
        ss = ss + f * f
    }
    ss = ss / Float(x.count)
    ss = ss + 1e-5
    ss = Float(1.0) / sqrt(ss)
    // normalize and scale
    for (i,v) in weights.enumerated() {
        out[i] = v * x[i] * ss
    }
}

func softmax(out: inout Array<Float>, startIndex: Int, size: Int) {
    // find max value (for numerical stability)
    var maxValue = out[0]
    
    for i in 1..<size {
        if out[i+startIndex] > maxValue {
            maxValue = out[i+startIndex]
        }
    }
    // exp and sum
    var sum:Float = 0.0
    for i in 0..<size {
        out[i+startIndex] = expf(out[i+startIndex] - maxValue)
        sum += out[i+startIndex]
    }
    // normalize
    for i in 0..<size {
        out[i+startIndex] /= sum
    }
}

func matmul(out: inout Array<Float>, x: Array<Float>, weights: ArraySlice<Float>) {
    //assert(out.count == x.count)
    for i in 0..<out.count {
        var val:Float = 0.0
        for j in 0..<x.count {
            val += x[j] * weights[(i * x.count + j)+weights.startIndex]
        }
        out[i] = val
    }
}

func transformer(token: Int, pos: Int, config:Config, weights: TransformerWeights, state: inout RunState) {
    let dim = config.dim
    let hiddenDim = config.hidden_dim
    let headSize = dim / config.n_heads
    
    // copy the token embedding into x
    state.x.copyFromSlice(slice: weights.token_embedding_table[token * dim ..< (token*dim)+dim])
    
    // pluck out the "pos" row of freq_cis_real and freq_cis_imag
    let freqCisRealRow = weights.freq_cis_real[pos * headSize / 2 ..< (pos * headSize / 2) + dim]
    let freqCisImaglRow = weights.freq_cis_imag[pos * headSize / 2 ..< (pos * headSize / 2) + dim]
    
    // forward all the layers
    for l in 0 ..< config.n_layers {
        
        // attention rmsnorm
        rmsnorn(out:&state.xb, x: state.x, weights: weights.rms_att_weight[l * dim ..< (l * dim) + dim])
        
        // qkv matmuls for this position
        let index1 = l * dim * dim
        let index2 = index1 + dim * dim
        matmul(out: &state.q, x: state.xb, weights: weights.wq[index1 ..< index2])
        matmul(out: &state.k, x: state.xb, weights: weights.wk[index1 ..< index2])
        matmul(out: &state.v, x: state.xb, weights: weights.wv[index1 ..< index2])
        
        // RoPE relative positional encoding: complex-valued rotate q and k by freq_cis in each head
        for i in stride(from: 0, to: dim, by: 2) {
            let q0 = state.q[i]
            let q1 = state.q[i+1]
            let k0 = state.k[i]
            let k1 = state.k[i+1]
            let fcr = freqCisRealRow[((i % headSize) / 2) + freqCisRealRow.startIndex]
            let fci = freqCisImaglRow[((i % headSize) / 2) + freqCisImaglRow.startIndex]
            state.q[i] = q0 * fcr - q1 * fci
            state.q[i+1] = q0 * fci + q1 * fcr
            state.k[i] = k0 * fcr - k1 * fci
            state.k[i+1] = k0 * fci + k1 * fcr
        }
        
        // save key,value at this time step (pos) to our kv cache
        let loff = l * config.seq_len * dim // kv cache layer offset for convenience
        for (i, v) in state.k.enumerated() {
            state.keyCache[loff + pos*dim + i] = v
        }
        for  (i, v) in state.v.enumerated() {
            state.valueCache[loff + pos*dim + i] = v
        }
        
        // multihead attention. iterate over all heads
        for h in 0 ..< config.n_heads {
            // get the query index for this head
            let qIndex = h * headSize
            // attention scores index for this head
            let attIndex = h * config.seq_len
            // iterate over all timesteps, including the current one
            for t in 0...pos {
                // get the key vector index for this head and at this timestep
                let kIndex = loff + t*dim + h*headSize
                // calculate the attention score as the dot product of q and k
                var score: Float = 0.0
                for i in 0..<headSize {
                    score += state.q[qIndex+i] * state.keyCache[kIndex+i]
                }
                score = score / sqrtf(Float(headSize))
                // save the score to the attention buffer
                state.att[attIndex+t] = score
            }
            
            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(out: &state.att, startIndex: attIndex, size: pos + 1)
            
            // weighted sum of the values, store back into xb
            let xbIndex = h * headSize
            for i in 0..<headSize {
                state.xb[xbIndex+i] = 0
            }
            for t in 0...pos {
                // get the value vector index for this head and at this timestep
                let vIndex = loff + t*dim + h*headSize
                // get the attention weight for this timestep
                let a = state.att[attIndex+t]
                // accumulate the weighted value into xb
                for i in 0..<headSize {
                    state.xb[xbIndex+i] += state.valueCache[vIndex+i] * a
                }
            }
        } // iterration over heads
        
        // final matmul to get the output of the attention
        matmul(out: &state.xb2, x: state.xb, weights: weights.wo[index1 ..< index2])
        
        // residual connection back into x
        accum(out: &state.x, b: state.xb2)
        
        // ffn rmsnorm
        rmsnorn(out: &state.xb, x: state.x, weights: weights.rms_ffn_weight[l*dim..<l*dim+dim])
        
        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        let wIndex1 = l*dim*hiddenDim
        let wIndex2 = wIndex1 + dim*hiddenDim
        matmul(out: &state.hb, x: state.xb, weights: weights.w1[wIndex1..<wIndex2])
        matmul(out: &state.hb2, x: state.xb, weights: weights.w3[wIndex1..<wIndex2])
        
        // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
        for i in 0..<hiddenDim {
            state.hb[i] = state.hb[i] * (Float(1.0) / (Float(1.0) + expf(-state.hb[i])))
        }
        
        // elementwise multiply with w3(x)
        for i in 0..<hiddenDim {
            state.hb[i] = state.hb[i] * state.hb2[i]
        }
        
        // final matmul to get the output of the ffn
        matmul(out: &state.xb, x: state.hb, weights: weights.w2[wIndex1..<wIndex2])
        
        // residual connection
        accum(out: &state.x, b: state.xb)
    } // iterration over layers
    
    // final rmsnorm
    rmsnorn(out: &state.x, x: state.x, weights: weights.rms_final_weight[0..<dim])
    
    // classifier into logits
    matmul(out: &state.logits, x: state.x, weights: weights.wcls[0..<weights.wcls.count])
}

// MARK: - Sampling
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

struct Sampling {
    var rng_seed: UInt64
    
    init(rng_seed: UInt64) {
        if rng_seed == 0 {
            self.rng_seed = UInt64.random(in: 0..<UInt64.max)
        } else {
            self.rng_seed = rng_seed
        }
    }
    
    func argmax(_ array: [Float]) -> Int {
        var maxIndex: Int = 0
        var maxProb: Float = array[0]
        for (index, prob) in array.enumerated() {
            if prob > maxProb {
                maxIndex = index
                maxProb = prob
            }
        }
        return maxIndex
    }
    
    mutating func sample(_ array: [Float]) -> Int {
        // sample index from probabilities (they must sum to 1!)
        let r = randomF32()
        var cdf:Float = 0.0
        for (index, prob) in array.enumerated() {
            cdf += prob
            if r < cdf {
                return index
            }
        }
        return array.count - 1
    }
    
    mutating func randomU32() -> UInt32 {
        rng_seed ^= rng_seed >> 12
        rng_seed ^= rng_seed << 25
        rng_seed ^= rng_seed >> 27
        var t = UInt64(rng_seed &* 0x2545F4914F6CDD1)
        t = t >> 32
        return UInt32(truncatingIfNeeded: t)
    }
    
    mutating func randomF32() -> Float {
        return Float(Float(randomU32() >> 8) / Float(16777216.0))
    }
    
    mutating func sampleTopP(_ array: [Float], top: Float) -> Int {
        var probindex = Array(repeating: ProbIndex(), count: array.count)
        for (index, prob) in array.enumerated() {
            probindex[index].prob = prob
            probindex[index].index = index
        }
        
        probindex.sort{
            $0.prob > $1.prob
        }
        
        // truncate the list where cumulative probability exceeds topp
        var cumulativeProb:Float = 0.0
        var lastIndex: Int = 0
        for i in 0..<array.count {
            cumulativeProb += probindex[i].prob
            if cumulativeProb > top {
                lastIndex = i // we've exceeded topp by including last_idx
                break
            }
        }
        
        // sample from the truncated list
        let r:Float = randomF32() * cumulativeProb
        var cdf: Float = 0.0
        for i in 0...lastIndex {
            cdf += probindex[i].prob
            if r < cdf {
                return probindex[i].index
            }
        }
        return probindex[lastIndex].index
    }
}

struct Arguments: ParsableArguments {
    @Argument(help: "model checkpoint path")
    var checkpoint: String = ""
    
    @Option(name: [.customShort("t")], help: "temperature")
    var temperature: Float = 1.0
    
    @Option(name: [.customShort("p")], help: "value in top-p (nucleus) sampling, 0 = off")
    var topp: Float = 0.9
    
    @Option(name: [.customShort("s")], help: "random seed, 0 = random")
    var seed: UInt64 = 0
    
    @Option(name: [.customShort("n")], help: "number of steps to run for, 0 = max_seq_len")
    var steps: Int = 256;
    
    @Option(name: [.customShort("i")], help: "input prompt")
    var prompt: String = "Once upon a time";
}

let args = Arguments.parseOrExit()

var sampling = Sampling(rng_seed: args.seed)

// MARK: - MAIN

// (1) Load Weights
guard let url = URL(string: args.checkpoint) else {
    print("Invalid chekpoint path: \(args.checkpoint)")
    exit(1)
}
guard let file = try? FileHandle(forReadingFrom: url) else {
    print("Can't read checkpoints")
    exit(2)
}

guard let config = Config(with: file) else {
    print("Can't init config")
    exit(3)
}

guard let transformerWeights = TransformerWeights(with: file, config: config) else {
    print("Can't init weights")
    exit(4)
}

// (2) Load vocab
var tokenizer = BPETokenizer()
tokenizer.loadVocab(fileName: "tokenizer.bin", vocabSize: config.vocab_size)


// (3) initalize
let temperature = args.temperature // 0.0 = greedy deterministic. 1.0 = original. don't set higher
let topp = args.topp // top-p in nucleus sampling
let steps: Int // number of steps to run for

// right now we cannot run for more than config.seq_len steps
if (args.steps <= 0 || args.steps > config.seq_len) {
    steps = config.seq_len;
} else {
    steps = args.steps
}

print(steps)

// (4) process the promp
let promptTokens = tokenizer.bpeEncode(args.prompt)

// (5) start the main loop
var next: Int = 0 // will store the next token in the sequence
var token: Int = 1 // init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
var pos: Int = 0 // position in the sequence
var state = RunState(config: config)

// used to time our code, only initialized after first iteration
var start: CFTimeInterval = 0

while (pos < steps) {
    // forward the transformer to get logits for the next token
    transformer(token:token, pos: pos, config: config, weights:transformerWeights, state:
                    &state)
    
    // advance the state machine
    if (pos < promptTokens.count) {
        // if we are still processing the input prompt, force the next prompt token
        next = promptTokens[pos]
    } else {
        // sample the next token
        if (temperature == 0.0) {
            // greedy argmax sampling: take the token with the highest probability
            next = sampling.argmax(state.logits)
        } else {
            // apply the temperature to the logits
            for i in 0..<state.logits.count {
                state.logits[i] /= temperature
            }
            // apply softmax to the logits to get the probabilities for next token
            softmax(out: &state.logits, startIndex:0, size: state.logits.count)
            // we sample from this distribution to get the next token
            if (topp <= 0) {
                // simply sample from the predicted probability distribution
                next = sampling.sample(state.logits)
            } else {
                // top-p (nucleus) sampling, clamping the least likely tokens to zero
                next = sampling.sampleTopP(state.logits, top: topp)
            }
        }
    }
    pos += 1
    
    // data-dependent terminating condition: the BOS (1) token delimits sequences
    if next == 1 {
        break
    }
    
    let str = tokenizer.vocab[next]
    print(str, terminator: "")
    
    token = next
    
    if start == 0 {
        start = CFAbsoluteTimeGetCurrent()
    }
}

if pos > 1 {
    let elapsed = CFAbsoluteTimeGetCurrent() - start
    print("\n\nAchieved tok/s: \(Double((pos-1))/elapsed) seconds\n")
}