import Foundation
import Vision
import AppKit

enum OCRToolError: Error {
    case invalidArguments
    case imageLoadFailed
}

func loadImage(path: String) throws -> CGImage {
    guard let image = NSImage(contentsOfFile: path) else {
        throw OCRToolError.imageLoadFailed
    }
    var rect = NSRect(origin: .zero, size: image.size)
    guard let cgImage = image.cgImage(forProposedRect: &rect, context: nil, hints: nil) else {
        throw OCRToolError.imageLoadFailed
    }
    return cgImage
}

func recognizeText(in path: String) throws -> String {
    let cgImage = try loadImage(path: path)
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    request.recognitionLanguages = ["zh-Hans", "en-US"]
    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    try handler.perform([request])
    let observations = request.results ?? []
    let lines = observations.compactMap { observation in
        observation.topCandidates(1).first?.string.trimmingCharacters(in: .whitespacesAndNewlines)
    }.filter { !$0.isEmpty }
    return lines.joined(separator: "\n")
}

do {
    let arguments = CommandLine.arguments
    guard arguments.count == 2 else {
        throw OCRToolError.invalidArguments
    }
    let text = try recognizeText(in: arguments[1])
    FileHandle.standardOutput.write(Data(text.utf8))
} catch {
    let message = String(describing: error)
    FileHandle.standardError.write(Data(message.utf8))
    exit(1)
}
