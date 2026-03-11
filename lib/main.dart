import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart';


Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MnistApp());
}

enum DemoStage {
  draw,
  preprocess,
  network,
  result,
}

class MnistApp extends StatelessWidget {
  const MnistApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'MNIST Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF0F766E),
          brightness: Brightness.dark,
        ),
        scaffoldBackgroundColor: const Color(0xFF020617),
        useMaterial3: true,
      ),
      home: const MnistHomePage(),
    );
  }
}

class MnistHomePage extends StatefulWidget {
  const MnistHomePage({super.key});

  @override
  State<MnistHomePage> createState() => _MnistHomePageState();
}

class _MnistHomePageState extends State<MnistHomePage>
    with SingleTickerProviderStateMixin {
  static const double canvasSize = 280;

  static const double expandedCanvasMaxSize = 720;

  double get _drawCanvasSize => canvasSize;
  static const int mnistSize = 28;
  static const double targetDigitSize = 20;
  static const double inkThreshold = 0.18;

  final GlobalKey _repaintKey = GlobalKey();
  final List<List<Offset>> _strokes = <List<Offset>>[];
  final ValueNotifier<int> _strokeRevision = ValueNotifier<int>(0);

  late final AnimationController _networkController;

  MnistDenseModel? _model;
  MnistProcessingResult? _processingResult;
  MnistInferenceResult? _inferenceResult;
  DemoStage _stage = DemoStage.draw;
  bool _predicting = false;
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    _networkController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 3200),
    );
    unawaited(_loadModel());
  }

  @override
  void dispose() {
    _networkController.dispose();
    _strokeRevision.dispose();
    super.dispose();
  }

  void _notifyStrokeListeners() {
    _strokeRevision.value++;
  }

  Future<void> _loadModel() async {
    try {
      final String raw = await rootBundle.loadString('assets/models/mnist_dense.json');
      final Map<String, dynamic> json = jsonDecode(raw) as Map<String, dynamic>;
      final MnistDenseModel model = MnistDenseModel.fromJson(json);
      if (!mounted) {
        return;
      }
      setState(() {
        _model = model;
      });
    } catch (_) {
      if (!mounted) {
        return;
      }
      setState(() {
        _errorMessage =
            'Modell konnte nicht geladen werden. Fuehre `python tool/train_mnist.py` neu aus.';
      });
    }
  }

  void _openCanvasOverlay() {
    showGeneralDialog<void>(
      context: context,
      barrierDismissible: true,
      barrierLabel: 'Canvas Overlay',
      barrierColor: const Color(0xE6000000),
      transitionDuration: const Duration(milliseconds: 180),
      pageBuilder: (BuildContext context, Animation<double> animation, Animation<double> secondaryAnimation) {
        final double overlaySize = _overlayCanvasSize(context);
        return SafeArea(
          child: Center(
            child: Container(
              width: overlaySize + 40,
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: const Color(0xFF0B1220),
                borderRadius: BorderRadius.circular(28),
                boxShadow: const <BoxShadow>[
                  BoxShadow(
                    color: Color(0x66000000),
                    blurRadius: 40,
                    offset: Offset(0, 20),
                  ),
                ],
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: <Widget>[
                  Row(
                    children: <Widget>[
                      const Text(
                        'Zeichenflaeche',
                        style: TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.w700),
                      ),
                      const Spacer(),
                      IconButton(
                        tooltip: 'Schliessen',
                        onPressed: () => Navigator.of(context).pop(),
                        icon: const Icon(Icons.close, color: Colors.white),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  _buildInteractiveCanvas(
                    overlaySize,
                    showExpandButton: false,
                    useRepaintBoundary: false,
                  ),
                ],
              ),
            ),
          ),
        );
      },
      transitionBuilder: (BuildContext context, Animation<double> animation, Animation<double> secondaryAnimation, Widget child) {
        return FadeTransition(
          opacity: animation,
          child: ScaleTransition(
            scale: Tween<double>(begin: 0.96, end: 1.0).animate(animation),
            child: child,
          ),
        );
      },
    );
  }
  double _overlayCanvasSize(BuildContext context) {
    final Size size = MediaQuery.of(context).size;
    return math.min(math.min(size.width - 120, size.height - 180), expandedCanvasMaxSize).toDouble();
  }

  Offset? _normalizePosition(Offset position, double displaySize) {
    if (position.dx < 0 ||
        position.dy < 0 ||
        position.dx > displaySize ||
        position.dy > displaySize) {
      return null;
    }
    return Offset(
      (position.dx.clamp(0, displaySize) / displaySize) * canvasSize,
      (position.dy.clamp(0, displaySize) / displaySize) * canvasSize,
    );
  }

  void _startStroke(Offset position, double displaySize) {
    final Offset? normalized = _normalizePosition(position, displaySize);
    if (normalized == null) {
      return;
    }
    setState(() {
      _strokes.add(<Offset>[normalized]);
      _processingResult = null;
      _inferenceResult = null;
      _errorMessage = null;
    });
    _notifyStrokeListeners();
  }
  void _appendStroke(Offset position, double displaySize) {
    final Offset? normalized = _normalizePosition(position, displaySize);
    if (normalized == null || _strokes.isEmpty) {
      return;
    }
    _strokes.last.add(normalized);
    _notifyStrokeListeners();
  }
  void _resetDemo() {
    _networkController.reset();
    setState(() {
      _strokes.clear();
      _processingResult = null;
      _inferenceResult = null;
      _stage = DemoStage.draw;
      _errorMessage = null;
    });
    _notifyStrokeListeners();
  }
  Future<void> _analyzeDrawing() async {
    if (_model == null || _strokes.isEmpty) {
      return;
    }

    setState(() {
      _predicting = true;
      _errorMessage = null;
    });

    try {
      final RenderRepaintBoundary boundary =
          _repaintKey.currentContext!.findRenderObject()! as RenderRepaintBoundary;
      final ui.Image image = await boundary.toImage(pixelRatio: 1);
      final ByteData? byteData = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
      if (byteData == null) {
        throw Exception('Keine Bilddaten erzeugt.');
      }

      final MnistProcessingResult processing = _processToMnist(
        byteData.buffer.asUint8List(),
        image.width,
        image.height,
      );
      final MnistInferenceResult inference = _model!.infer(processing.centeredPixels);

      if (!mounted) {
        return;
      }
      setState(() {
        _processingResult = processing;
        _inferenceResult = inference;
        _stage = DemoStage.preprocess;
      });
    } catch (_) {
      if (!mounted) {
        return;
      }
      setState(() {
        _errorMessage = 'Analyse fehlgeschlagen. Erzeuge das Modell bitte neu.';
      });
    } finally {
      if (mounted) {
        setState(() {
          _predicting = false;
        });
      }
    }
  }

  Future<void> _goNext() async {
    if (_stage == DemoStage.draw) {
      await _analyzeDrawing();
      return;
    }

    if (_stage == DemoStage.preprocess) {
      _networkController.forward(from: 0);
      setState(() {
        _stage = DemoStage.network;
      });
      return;
    }

    if (_stage == DemoStage.network) {
      setState(() {
        _stage = DemoStage.result;
      });
    }
  }

  void _goBack() {
    if (_stage == DemoStage.preprocess) {
      setState(() {
        _stage = DemoStage.draw;
      });
      return;
    }

    if (_stage == DemoStage.network) {
      _networkController.reset();
      setState(() {
        _stage = DemoStage.preprocess;
      });
      return;
    }

    if (_stage == DemoStage.result) {
      _networkController.forward(from: 0.7);
      setState(() {
        _stage = DemoStage.network;
      });
    }
  }

  MnistProcessingResult _processToMnist(Uint8List rgbaBytes, int width, int height) {
    final List<double> grayscale = List<double>.filled(width * height, 0);
    int minX = width;
    int minY = height;
    int maxX = -1;
    int maxY = -1;

    for (int i = 0; i < width * height; i++) {
      final int base = i * 4;
      final double r = rgbaBytes[base].toDouble();
      final double g = rgbaBytes[base + 1].toDouble();
      final double b = rgbaBytes[base + 2].toDouble();
      final double luminance = (0.299 * r) + (0.587 * g) + (0.114 * b);
      double value = 1 - (luminance / 255.0);
      if (value < inkThreshold) {
        value = 0;
      }
      grayscale[i] = value;

      if (value > 0) {
        final int x = i % width;
        final int y = i ~/ width;
        minX = math.min(minX, x);
        minY = math.min(minY, y);
        maxX = math.max(maxX, x);
        maxY = math.max(maxY, y);
      }
    }

    final List<double> rawDownsampled = _downsample(grayscale, width, height, mnistSize, mnistSize);

    if (maxX < minX || maxY < minY) {
      return MnistProcessingResult(
        rawPixels: rawDownsampled,
        scaledPixels: List<double>.filled(mnistSize * mnistSize, 0),
        centeredPixels: List<double>.filled(mnistSize * mnistSize, 0),
        boundingWidth: 0,
        boundingHeight: 0,
        scaleFactor: 1,
        shiftX: 0,
        shiftY: 0,
      );
    }

    final int boxWidth = maxX - minX + 1;
    final int boxHeight = maxY - minY + 1;
    final double scale = targetDigitSize / math.max(boxWidth, boxHeight);
    final int resizedWidth = math.max(1, (boxWidth * scale).round());
    final int resizedHeight = math.max(1, (boxHeight * scale).round());
    final List<double> resized = List<double>.filled(resizedWidth * resizedHeight, 0);

    for (int y = 0; y < resizedHeight; y++) {
      for (int x = 0; x < resizedWidth; x++) {
        final double srcX = minX + ((x + 0.5) / scale) - 0.5;
        final double srcY = minY + ((y + 0.5) / scale) - 0.5;
        resized[(y * resizedWidth) + x] = _sampleBilinear(grayscale, width, height, srcX, srcY);
      }
    }

    final List<double> scaledCanvas = List<double>.filled(mnistSize * mnistSize, 0);
    final int offsetX = ((mnistSize - resizedWidth) / 2).floor();
    final int offsetY = ((mnistSize - resizedHeight) / 2).floor();

    for (int y = 0; y < resizedHeight; y++) {
      for (int x = 0; x < resizedWidth; x++) {
        final int destX = offsetX + x;
        final int destY = offsetY + y;
        if (destX >= 0 && destX < mnistSize && destY >= 0 && destY < mnistSize) {
          scaledCanvas[(destY * mnistSize) + destX] = resized[(y * resizedWidth) + x];
        }
      }
    }

    final CenteredPixelsResult centered = _centerByMass(scaledCanvas);
    return MnistProcessingResult(
      rawPixels: rawDownsampled,
      scaledPixels: scaledCanvas,
      centeredPixels: centered.pixels,
      boundingWidth: boxWidth,
      boundingHeight: boxHeight,
      scaleFactor: scale,
      shiftX: centered.shiftX,
      shiftY: centered.shiftY,
    );
  }

  List<double> _downsample(
    List<double> pixels,
    int width,
    int height,
    int targetWidth,
    int targetHeight,
  ) {
    final List<double> result = List<double>.filled(targetWidth * targetHeight, 0);
    final double scaleX = width / targetWidth;
    final double scaleY = height / targetHeight;

    for (int y = 0; y < targetHeight; y++) {
      for (int x = 0; x < targetWidth; x++) {
        double sum = 0;
        int count = 0;
        final int startX = (x * scaleX).floor();
        final int endX = math.max(startX + 1, ((x + 1) * scaleX).ceil());
        final int startY = (y * scaleY).floor();
        final int endY = math.max(startY + 1, ((y + 1) * scaleY).ceil());

        for (int py = startY; py < endY && py < height; py++) {
          for (int px = startX; px < endX && px < width; px++) {
            sum += pixels[(py * width) + px];
            count++;
          }
        }

        result[(y * targetWidth) + x] = count == 0 ? 0 : sum / count;
      }
    }

    return result;
  }

  double _sampleBilinear(List<double> pixels, int width, int height, double x, double y) {
    final double clampedX = x.clamp(0, width - 1).toDouble();
    final double clampedY = y.clamp(0, height - 1).toDouble();
    final int x0 = clampedX.floor();
    final int y0 = clampedY.floor();
    final int x1 = math.min(x0 + 1, width - 1);
    final int y1 = math.min(y0 + 1, height - 1);
    final double dx = clampedX - x0;
    final double dy = clampedY - y0;

    final double topLeft = pixels[(y0 * width) + x0];
    final double topRight = pixels[(y0 * width) + x1];
    final double bottomLeft = pixels[(y1 * width) + x0];
    final double bottomRight = pixels[(y1 * width) + x1];

    final double top = (topLeft * (1 - dx)) + (topRight * dx);
    final double bottom = (bottomLeft * (1 - dx)) + (bottomRight * dx);
    return (top * (1 - dy)) + (bottom * dy);
  }

  CenteredPixelsResult _centerByMass(List<double> pixels) {
    double totalMass = 0;
    double sumX = 0;
    double sumY = 0;

    for (int y = 0; y < mnistSize; y++) {
      for (int x = 0; x < mnistSize; x++) {
        final double value = pixels[(y * mnistSize) + x];
        totalMass += value;
        sumX += x * value;
        sumY += y * value;
      }
    }

    if (totalMass == 0) {
      return CenteredPixelsResult(
        pixels: pixels,
        shiftX: 0,
        shiftY: 0,
      );
    }

    final double centerX = sumX / totalMass;
    final double centerY = sumY / totalMass;
    final int shiftX = (13.5 - centerX).round();
    final int shiftY = (13.5 - centerY).round();

    if (shiftX == 0 && shiftY == 0) {
      return CenteredPixelsResult(
        pixels: pixels,
        shiftX: 0,
        shiftY: 0,
      );
    }

    final List<double> shifted = List<double>.filled(mnistSize * mnistSize, 0);
    for (int y = 0; y < mnistSize; y++) {
      for (int x = 0; x < mnistSize; x++) {
        final int sourceIndex = (y * mnistSize) + x;
        final int targetX = x + shiftX;
        final int targetY = y + shiftY;
        if (targetX >= 0 && targetX < mnistSize && targetY >= 0 && targetY < mnistSize) {
          final int targetIndex = (targetY * mnistSize) + targetX;
          shifted[targetIndex] = math.max(shifted[targetIndex], pixels[sourceIndex]);
        }
      }
    }

    return CenteredPixelsResult(
      pixels: shifted,
      shiftX: shiftX,
      shiftY: shiftY,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Stack(
          children: <Widget>[
            Center(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 1320),
                child: Padding(
                  padding: const EdgeInsets.all(24),
                  child: Column(
                    children: <Widget>[
                      _buildHeader(context),
                      const SizedBox(height: 20),
                      Expanded(
                        child: AnimatedSwitcher(
                          duration: const Duration(milliseconds: 220),
                          child: KeyedSubtree(
                            key: ValueKey<DemoStage>(_stage),
                            child: _buildStageBody(context),
                          ),
                        ),
                      ),
                      const SizedBox(height: 20),
                      _buildNavigationBar(),
                    ],
                  ),
                ),
              ),
            ),
            Positioned(
              right: 24,
              bottom: 16,
              child: IgnorePointer(
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                  decoration: BoxDecoration(
                    color: const Color(0xCC0F172A),
                    borderRadius: BorderRadius.circular(999),
                    border: Border.all(color: const Color(0xFF1E293B)),
                  ),
                  child: const Text(
                    'by Luca Rietsch',
                    style: TextStyle(
                      color: Color(0xFFCBD5E1),
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                      letterSpacing: 0.3,
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: <Widget>[
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: <Widget>[
              Text(
                'MNIST Live Demo',
                style: Theme.of(context).textTheme.headlineLarge?.copyWith(
                      fontWeight: FontWeight.w800,
                    ),
              ),
              const SizedBox(height: 8),
              const Text(
                'Zeichnen, Vorverarbeitung, Netzwerk und Ergebnis als interaktive Schritt-fuer-Schritt-Ansicht.',
                style: TextStyle(fontSize: 16, color: Color(0xFF94A3B8)),
              ),
              const SizedBox(height: 16),
              Wrap(
                spacing: 12,
                runSpacing: 12,
                children: DemoStage.values.map((DemoStage stage) {
                  return StageChip(
                    number: stage.index + 1,
                    title: _stageTitle(stage),
                    active: _stage == stage,
                    done: stage.index < _stage.index,
                  );
                }).toList(),
              ),
            ],
          ),
        ),
        const SizedBox(width: 16),
        Wrap(
          spacing: 12,
          runSpacing: 12,
          children: <Widget>[

            FilledButton.tonalIcon(
              onPressed: _resetDemo,
              icon: const Icon(Icons.restart_alt),
              label: const Text('Reset'),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildNavigationBar() {
    final bool canAdvanceFromDraw = _model != null && _strokes.isNotEmpty && !_predicting;
    final bool canGoNext = switch (_stage) {
      DemoStage.draw => canAdvanceFromDraw,
      DemoStage.preprocess => true,
      DemoStage.network => true,
      DemoStage.result => false,
    };

    return Row(
      children: <Widget>[
        OutlinedButton.icon(
          onPressed: _stage == DemoStage.draw ? null : _goBack,
          icon: const Icon(Icons.arrow_back),
          label: const Text('Zurueck'),
        ),
        const Spacer(),
        if (_errorMessage != null)
          Expanded(
            child: Text(
              _errorMessage!,
              textAlign: TextAlign.center,
              style: const TextStyle(color: Color(0xFFB91C1C), fontWeight: FontWeight.w600),
            ),
          ),
        if (_errorMessage != null) const Spacer(),
        FilledButton.icon(
          onPressed: canGoNext ? _goNext : null,
          icon: _predicting
              ? const SizedBox(
                  width: 18,
                  height: 18,
                  child: CircularProgressIndicator(strokeWidth: 2),
                )
              : const Icon(Icons.arrow_forward),
          label: Text(_nextLabel()),
        ),
      ],
    );
  }

  Widget _buildStageBody(BuildContext context) {
    switch (_stage) {
      case DemoStage.draw:
        return _buildDrawStage(context);
      case DemoStage.preprocess:
        return _buildPreprocessStage();
      case DemoStage.network:
        return _buildNetworkStage(context);
      case DemoStage.result:
        return _buildResultStage();
    }
  }

  Widget _buildDrawStage(BuildContext context) {
    return Center(
      child: SizedBox(
        width: 620,
        child: _buildDrawingCard(context),
      ),
    );
  }

  Widget _buildDrawingCard(BuildContext context) {
    return Card(
      elevation: 0,
      color: const Color(0xFF0F172A),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(28)),
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Text(
              '1. Zahl zeichnen',
              style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                    fontWeight: FontWeight.w700,
                  ),
            ),
            const SizedBox(height: 10),
            const Text(
              'Zeichne eine Ziffer von 0 bis 9. Im Vollbild vergroessert sich nur das Zeichenfeld fuer eine sauberere Praesentation.',
              style: TextStyle(fontSize: 16, color: Color(0xFF94A3B8)),
            ),
            const SizedBox(height: 24),
            _buildInteractiveCanvas(
              _drawCanvasSize,
              showExpandButton: true,
              useRepaintBoundary: true,
            ),
            const SizedBox(height: 18),
            Wrap(
              spacing: 12,
              runSpacing: 12,
              children: <Widget>[
                FilledButton.icon(
                  onPressed: (_model != null && _strokes.isNotEmpty && !_predicting) ? _goNext : null,
                  icon: const Icon(Icons.slideshow),
                  label: const Text('Zur Demo weiter'),
                ),
                OutlinedButton.icon(
                  onPressed: _clearDrawingOnly,
                  icon: const Icon(Icons.refresh),
                  label: const Text('Feld leeren'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  void _clearDrawingOnly() {
    setState(() {
      _strokes.clear();
      _processingResult = null;
      _inferenceResult = null;
      _errorMessage = null;
    });
    _notifyStrokeListeners();
  }
  Widget _buildInteractiveCanvas(
    double displaySize, {
    required bool showExpandButton,
    required bool useRepaintBoundary,
  }) {
    Widget canvasSurface = Container(
      width: displaySize,
      height: displaySize,
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: const Color(0xFFB8CBC4), width: 2),
        boxShadow: const <BoxShadow>[
          BoxShadow(
            color: Color(0x33000000),
            blurRadius: 20,
            offset: Offset(0, 8),
          ),
        ],
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(22),
        child: GestureDetector(
          behavior: HitTestBehavior.opaque,
          onPanStart: (DragStartDetails details) {
            _startStroke(details.localPosition, displaySize);
          },
          onPanUpdate: (DragUpdateDetails details) {
            _appendStroke(details.localPosition, displaySize);
          },
          child: CustomPaint(
            painter: DigitPainter(
              strokes: _strokes,
              logicalCanvasSize: canvasSize,
              repaint: _strokeRevision,
            ),
            size: Size(displaySize, displaySize),
          ),
        ),
      ),
    );
    if (useRepaintBoundary) {
      canvasSurface = RepaintBoundary(
        key: _repaintKey,
        child: canvasSurface,
      );
    }
    return Padding(
      padding: EdgeInsets.only(
        top: showExpandButton ? 14 : 0,
        right: showExpandButton ? 14 : 0,
      ),
      child: Stack(
        clipBehavior: Clip.none,
        children: <Widget>[
          canvasSurface,
          if (showExpandButton)
            Positioned(
              top: -14,
              right: -14,
              child: Material(
                color: const Color(0xCC0F172A),
                shape: const CircleBorder(),
                elevation: 6,
                child: IconButton(
                  tooltip: 'Zeichenfeld vergroessern',
                  onPressed: _openCanvasOverlay,
                  icon: const Icon(Icons.fullscreen, color: Colors.white),
                ),
              ),
            ),
        ],
      ),
    );
  }
  Widget _buildPreprocessStage() {
    final MnistProcessingResult? processing = _processingResult;
    if (processing == null) {
      return const SizedBox.shrink();
    }

    return SingleChildScrollView(
      child: Wrap(
        spacing: 24,
        runSpacing: 24,
        children: <Widget>[
          PixelGridCard(
            title: '2.1 Roh auf 28x28',
            subtitle: 'So saehe die Zeichnung ohne Korrektur aus.',
            pixels: processing.rawPixels,
          ),
          PixelGridCard(
            title: '2.2 Skaliert',
            subtitle: 'Die Zahl wird zugeschnitten und auf etwa 20x20 gebracht.',
            pixels: processing.scaledPixels,
          ),
          PixelGridCard(
            title: '2.3 Zentriert',
            subtitle: 'Das finale Modell-Input. Hover ueber ein Pixel fuer den Wert.',
            pixels: processing.centeredPixels,
            interactive: true,
          ),
          SizedBox(
            width: 320,
            child: InfoCard(
              title: 'Vorverarbeitung',
              lines: <String>[
                'Bounding Box: ${processing.boundingWidth} x ${processing.boundingHeight} Pixel',
                'Skalierungsfaktor: ${processing.scaleFactor.toStringAsFixed(2)}',
                'Schwerpunkt-Verschiebung X: ${processing.shiftX}',
                'Schwerpunkt-Verschiebung Y: ${processing.shiftY}',
                'Das ist der Grund, warum auch grosse oder verschobene Ziffern stabiler erkannt werden.',
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildNetworkStage(BuildContext context) {
    final MnistInferenceResult? inference = _inferenceResult;
    if (inference == null) {
      return const SizedBox.shrink();
    }

    final List<int> topHidden1 = _topIndices(inference.hidden1, 3);
    final List<int> topHidden2 = _topIndices(inference.hidden2, 3);

    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Card(
            elevation: 0,
            color: const Color(0xFF0F172A),
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(28)),
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: <Widget>[
                  Text(
                    '3. Netzwerk-Berechnung',
                    style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                          fontWeight: FontWeight.w700,
                        ),
                  ),
                  const SizedBox(height: 10),
                  Text(
                    _networkPhaseText(),
                    style: const TextStyle(fontSize: 16, color: Color(0xFF94A3B8)),
                  ),
                  const SizedBox(height: 24),
                  AnimatedBuilder(
                    animation: _networkController,
                    builder: (BuildContext context, Widget? child) {
                      return CustomPaint(
                        painter: NetworkDiagramPainter(
                          progress: _networkController.value,
                          highlightedDigit: inference.prediction,
                        ),
                        child: const SizedBox(height: 360, width: double.infinity),
                      );
                    },
                  ),
                  const SizedBox(height: 16),
                  const Row(
                    children: <Widget>[
                      Expanded(child: LayerLabel(title: 'Input', count: '784 Neuronen')),
                      Expanded(child: LayerLabel(title: 'Hidden 1', count: '256 Neuronen')),
                      Expanded(child: LayerLabel(title: 'Hidden 2', count: '128 Neuronen')),
                      Expanded(child: LayerLabel(title: 'Output', count: '10 Neuronen')),
                    ],
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 24),
          Wrap(
            spacing: 24,
            runSpacing: 24,
            children: <Widget>[
              SizedBox(
                width: 320,
                child: InfoCard(
                  title: 'Aktivierungen',
                  lines: <String>[
                    'Starke Hidden-1-Neuronen: ${topHidden1.map((int i) => '#$i').join(', ')}',
                    'Starke Hidden-2-Neuronen: ${topHidden2.map((int i) => '#$i').join(', ')}',
                    'Die Animation zeigt vereinfacht, wie die Signale durch das Netz laufen.',
                  ],
                ),
              ),
              SizedBox(
                width: 320,
                child: PixelGridCard(
                  title: 'Eingabe ans Netzwerk',
                  subtitle: 'Das finale 28x28-Bild, das in die 784 Eingaben geht.',
                  pixels: _processingResult!.centeredPixels,
                  interactive: true,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildResultStage() {
    final MnistInferenceResult? inference = _inferenceResult;
    if (inference == null) {
      return const SizedBox.shrink();
    }

    final List<int> ranking = List<int>.generate(10, (int index) => index)
      ..sort((int a, int b) => inference.probabilities[b].compareTo(inference.probabilities[a]));

    return Wrap(
      spacing: 24,
      runSpacing: 24,
      children: <Widget>[
        SizedBox(
          width: 360,
          child: PixelGridCard(
            title: '4. Finale Eingabe',
            subtitle: 'Das Bild, auf dessen Basis die Vorhersage getroffen wurde.',
            pixels: _processingResult!.centeredPixels,
            interactive: true,
          ),
        ),
        SizedBox(
          width: 680,
          child: Card(
            elevation: 0,
            color: const Color(0xFF0F172A),
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(28)),
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: <Widget>[
                  const Text(
                    '4. Endergebnis',
                    style: TextStyle(color: Colors.white, fontSize: 32, fontWeight: FontWeight.w700),
                  ),
                  const SizedBox(height: 20),
                  Container(
                    padding: const EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      color: const Color(0xFF111827),
                      borderRadius: BorderRadius.circular(24),
                    ),
                    child: Row(
                      children: <Widget>[
                        Text(
                          '${inference.prediction}',
                          style: const TextStyle(
                            fontSize: 88,
                            height: 1,
                            color: Colors.white,
                            fontWeight: FontWeight.w800,
                          ),
                        ),
                        const SizedBox(width: 20),
                        Expanded(
                          child: Text(
                            '${(inference.probabilities[inference.prediction] * 100).toStringAsFixed(2)} % Wahrscheinlichkeit',
                            style: const TextStyle(
                              color: Color(0xFF86EFAC),
                              fontSize: 24,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 24),
                  ...ranking.map((int digit) {
                    final double probability = inference.probabilities[digit];
                    return Padding(
                      padding: const EdgeInsets.only(bottom: 12),
                      child: Row(
                        children: <Widget>[
                          SizedBox(
                            width: 24,
                            child: Text(
                              '$digit',
                              style: const TextStyle(color: Colors.white, fontSize: 16),
                            ),
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: ClipRRect(
                              borderRadius: BorderRadius.circular(999),
                              child: LinearProgressIndicator(
                                value: probability.clamp(0, 1),
                                minHeight: 18,
                                backgroundColor: const Color(0xFF334155),
                                valueColor: AlwaysStoppedAnimation<Color>(
                                  digit == inference.prediction
                                      ? const Color(0xFF34D399)
                                      : const Color(0xFF60A5FA),
                                ),
                              ),
                            ),
                          ),
                          const SizedBox(width: 12),
                          SizedBox(
                            width: 74,
                            child: Text(
                              '${(probability * 100).toStringAsFixed(1)} %',
                              textAlign: TextAlign.right,
                              style: const TextStyle(color: Color(0xFFCBD5E1), fontSize: 14),
                            ),
                          ),
                        ],
                      ),
                    );
                  }),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }

  String _stageTitle(DemoStage stage) {
    switch (stage) {
      case DemoStage.draw:
        return 'Zeichnen';
      case DemoStage.preprocess:
        return 'Vorverarbeitung';
      case DemoStage.network:
        return 'Netzwerk';
      case DemoStage.result:
        return 'Ergebnis';
    }
  }

  String _nextLabel() {
    switch (_stage) {
      case DemoStage.draw:
        return 'Weiter zur Vorverarbeitung';
      case DemoStage.preprocess:
        return 'Weiter zum Netzwerk';
      case DemoStage.network:
        return 'Weiter zum Ergebnis';
      case DemoStage.result:
        return 'Fertig';
    }
  }

  String _networkPhaseText() {
    final double value = _networkController.value;
    if (value < 0.33) {
      return 'Phase 1: Die 784 Eingabewerte werden in die erste Hidden-Layer eingespeist.';
    }
    if (value < 0.66) {
      return 'Phase 2: Die aktivierten Werte werden in die zweite Hidden-Layer weitergegeben.';
    }
    return 'Phase 3: Die Ausgaben fuer die Ziffern 0 bis 9 werden berechnet.';
  }

  List<int> _topIndices(List<double> values, int count) {
    final List<int> indices = List<int>.generate(values.length, (int index) => index)
      ..sort((int a, int b) => values[b].compareTo(values[a]));
    return indices.take(count).toList();
  }
}

class MnistProcessingResult {
  const MnistProcessingResult({
    required this.rawPixels,
    required this.scaledPixels,
    required this.centeredPixels,
    required this.boundingWidth,
    required this.boundingHeight,
    required this.scaleFactor,
    required this.shiftX,
    required this.shiftY,
  });

  final List<double> rawPixels;
  final List<double> scaledPixels;
  final List<double> centeredPixels;
  final int boundingWidth;
  final int boundingHeight;
  final double scaleFactor;
  final int shiftX;
  final int shiftY;
}

class CenteredPixelsResult {
  const CenteredPixelsResult({
    required this.pixels,
    required this.shiftX,
    required this.shiftY,
  });

  final List<double> pixels;
  final int shiftX;
  final int shiftY;
}

class MnistInferenceResult {
  const MnistInferenceResult({
    required this.hidden1,
    required this.hidden2,
    required this.logits,
    required this.probabilities,
    required this.prediction,
  });

  final List<double> hidden1;
  final List<double> hidden2;
  final List<double> logits;
  final List<double> probabilities;
  final int prediction;
}

class MnistDenseModel {
  MnistDenseModel({
    required this.hidden1Size,
    required this.hidden2Size,
    required this.outputSize,
    required this.dense1Kernel,
    required this.dense1Bias,
    required this.dense2Kernel,
    required this.dense2Bias,
    required this.dense3Kernel,
    required this.dense3Bias,
  });

  factory MnistDenseModel.fromJson(Map<String, dynamic> json) {
    return MnistDenseModel(
      hidden1Size: json['hidden1Size'] as int,
      hidden2Size: json['hidden2Size'] as int,
      outputSize: json['outputSize'] as int,
      dense1Kernel: (json['dense1Kernel'] as List<dynamic>)
          .map((dynamic value) => (value as num).toDouble())
          .toList(),
      dense1Bias: (json['dense1Bias'] as List<dynamic>)
          .map((dynamic value) => (value as num).toDouble())
          .toList(),
      dense2Kernel: (json['dense2Kernel'] as List<dynamic>)
          .map((dynamic value) => (value as num).toDouble())
          .toList(),
      dense2Bias: (json['dense2Bias'] as List<dynamic>)
          .map((dynamic value) => (value as num).toDouble())
          .toList(),
      dense3Kernel: (json['dense3Kernel'] as List<dynamic>)
          .map((dynamic value) => (value as num).toDouble())
          .toList(),
      dense3Bias: (json['dense3Bias'] as List<dynamic>)
          .map((dynamic value) => (value as num).toDouble())
          .toList(),
    );
  }

  final int hidden1Size;
  final int hidden2Size;
  final int outputSize;
  final List<double> dense1Kernel;
  final List<double> dense1Bias;
  final List<double> dense2Kernel;
  final List<double> dense2Bias;
  final List<double> dense3Kernel;
  final List<double> dense3Bias;

  MnistInferenceResult infer(List<double> input) {
    final List<double> hidden1 = List<double>.filled(hidden1Size, 0);
    for (int j = 0; j < hidden1Size; j++) {
      double sum = dense1Bias[j];
      for (int i = 0; i < input.length; i++) {
        sum += input[i] * dense1Kernel[(i * hidden1Size) + j];
      }
      hidden1[j] = math.max(0, sum);
    }

    final List<double> hidden2 = List<double>.filled(hidden2Size, 0);
    for (int j = 0; j < hidden2Size; j++) {
      double sum = dense2Bias[j];
      for (int i = 0; i < hidden1Size; i++) {
        sum += hidden1[i] * dense2Kernel[(i * hidden2Size) + j];
      }
      hidden2[j] = math.max(0, sum);
    }

    final List<double> logits = List<double>.filled(outputSize, 0);
    for (int j = 0; j < outputSize; j++) {
      double sum = dense3Bias[j];
      for (int i = 0; i < hidden2Size; i++) {
        sum += hidden2[i] * dense3Kernel[(i * outputSize) + j];
      }
      logits[j] = sum;
    }

    final List<double> probabilities = _softmax(logits);
    final int prediction = _argMax(probabilities);

    return MnistInferenceResult(
      hidden1: hidden1,
      hidden2: hidden2,
      logits: logits,
      probabilities: probabilities,
      prediction: prediction,
    );
  }

  List<double> _softmax(List<double> logits) {
    final double maxValue = logits.reduce(math.max);
    final List<double> exps = logits.map((double value) => math.exp(value - maxValue)).toList();
    final double sum = exps.reduce((double a, double b) => a + b);
    return exps.map((double value) => value / sum).toList();
  }

  int _argMax(List<double> values) {
    int bestIndex = 0;
    double bestValue = values.first;
    for (int i = 1; i < values.length; i++) {
      if (values[i] > bestValue) {
        bestValue = values[i];
        bestIndex = i;
      }
    }
    return bestIndex;
  }
}

class StageChip extends StatelessWidget {
  const StageChip({
    super.key,
    required this.number,
    required this.title,
    required this.active,
    required this.done,
  });

  final int number;
  final String title;
  final bool active;
  final bool done;

  @override
  Widget build(BuildContext context) {
    final Color background = active
        ? const Color(0xFF0F766E)
        : done
            ? const Color(0xFFDCFCE7)
            : Colors.white;
    final Color foreground = active
        ? Colors.white
        : done
            ? const Color(0xFF166534)
            : const Color(0xFF475569);

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: background,
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: active ? background : const Color(0xFFD7E1DE)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: <Widget>[
          CircleAvatar(
            radius: 13,
            backgroundColor: active ? Colors.white.withValues(alpha: 0.18) : const Color(0xFFF1F5F9),
            child: Text(
              '$number',
              style: TextStyle(
                color: foreground,
                fontSize: 12,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
          const SizedBox(width: 10),
          Text(
            title,
            style: TextStyle(color: foreground, fontWeight: FontWeight.w700),
          ),
        ],
      ),
    );
  }
}

class InfoCard extends StatelessWidget {
  const InfoCard({super.key, required this.title, required this.lines});

  final String title;
  final List<String> lines;

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 0,
      color: const Color(0xFF0F172A),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(28)),
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Text(
              title,
              style: Theme.of(context).textTheme.titleLarge?.copyWith(
                    fontWeight: FontWeight.w700,
                  ),
            ),
            const SizedBox(height: 16),
            ...lines.map(
              (String line) => Padding(
                padding: const EdgeInsets.only(bottom: 12),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: <Widget>[
                    const Padding(
                      padding: EdgeInsets.only(top: 6),
                      child: Icon(Icons.circle, size: 8, color: Color(0xFF0F766E)),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        line,
                        style: const TextStyle(fontSize: 15, color: Color(0xFFCBD5E1)),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class PixelGridCard extends StatefulWidget {
  const PixelGridCard({
    super.key,
    required this.title,
    required this.subtitle,
    required this.pixels,
    this.interactive = false,
  });

  final String title;
  final String subtitle;
  final List<double> pixels;
  final bool interactive;

  @override
  State<PixelGridCard> createState() => _PixelGridCardState();
}

class _PixelGridCardState extends State<PixelGridCard> {
  int? _hoveredIndex;

  @override
  Widget build(BuildContext context) {
    final int? hoveredIndex = _hoveredIndex;
    final String hoverText = hoveredIndex == null
        ? 'Hover ueber ein Pixel.'
        : 'Pixel (${hoveredIndex % 28}, ${hoveredIndex ~/ 28}) = ${widget.pixels[hoveredIndex].toStringAsFixed(3)}';

    return Card(
      elevation: 0,
      color: const Color(0xFF0F172A),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(28)),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Text(
              widget.title,
              style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 8),
            Text(widget.subtitle, style: const TextStyle(color: Color(0xFF94A3B8))),
            const SizedBox(height: 12),
            if (widget.interactive)
              Text(
                hoverText,
                style: const TextStyle(fontSize: 13, color: Color(0xFF0F766E), fontWeight: FontWeight.w600),
              )
            else
              const SizedBox(height: 20),
            const SizedBox(height: 12),
            SizedBox.square(
              dimension: 308,
              child: GridView.builder(
                physics: const NeverScrollableScrollPhysics(),
                gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                  crossAxisCount: 28,
                  crossAxisSpacing: 1,
                  mainAxisSpacing: 1,
                ),
                itemCount: 28 * 28,
                itemBuilder: (BuildContext context, int index) {
                  final double value = widget.pixels.length > index ? widget.pixels[index].clamp(0, 1) : 0;
                  final Color color = Color.lerp(Colors.white, Colors.black, value)!;
                  return MouseRegion(
                    onEnter: widget.interactive
                        ? (_) {
                            setState(() {
                              _hoveredIndex = index;
                            });
                          }
                        : null,
                    onExit: widget.interactive
                        ? (_) {
                            setState(() {
                              _hoveredIndex = null;
                            });
                          }
                        : null,
                    child: Container(
                      decoration: BoxDecoration(
                        color: color,
                        border: Border.all(
                          color: hoveredIndex == index ? const Color(0xFF0F766E) : const Color(0xFFE2E8F0),
                          width: hoveredIndex == index ? 1.4 : 0.35,
                        ),
                      ),
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class LayerLabel extends StatelessWidget {
  const LayerLabel({super.key, required this.title, required this.count});

  final String title;
  final String count;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: <Widget>[
        Text(title, style: const TextStyle(fontWeight: FontWeight.w700)),
        const SizedBox(height: 4),
        Text(count, style: const TextStyle(color: Color(0xFF64748B))),
      ],
    );
  }
}

class NetworkDiagramPainter extends CustomPainter {
  const NetworkDiagramPainter({required this.progress, required this.highlightedDigit});

  final double progress;
  final int highlightedDigit;

  @override
  void paint(Canvas canvas, Size size) {
    final List<Offset> inputNodes = _buildColumn(size, size.width * 0.10, 8, topPadding: 36, bottomPadding: 36);
    final List<Offset> hidden1Nodes = _buildColumn(size, size.width * 0.38, 8, topPadding: 24, bottomPadding: 24);
    final List<Offset> hidden2Nodes = _buildColumn(size, size.width * 0.64, 6, topPadding: 48, bottomPadding: 48);
    final List<Offset> outputNodes = _buildColumn(size, size.width * 0.90, 10, topPadding: 16, bottomPadding: 16);

    final Paint linePaint1 = Paint()
      ..color = const Color(0x550EA5E9)
      ..strokeWidth = 1.2;
    final Paint linePaint2 = Paint()
      ..color = const Color(0x554F46E5)
      ..strokeWidth = 1.2;
    final Paint linePaint3 = Paint()
      ..color = const Color(0x5534D399)
      ..strokeWidth = 1.2;

    final double phase1 = (progress / 0.33).clamp(0.0, 1.0);
    final double phase2 = ((progress - 0.33) / 0.33).clamp(0.0, 1.0);
    final double phase3 = ((progress - 0.66) / 0.34).clamp(0.0, 1.0);

    _drawConnections(canvas, inputNodes, hidden1Nodes, phase1, linePaint1);
    _drawConnections(canvas, hidden1Nodes, hidden2Nodes, phase2, linePaint2);
    _drawConnections(canvas, hidden2Nodes, outputNodes, phase3, linePaint3);

    _drawNodes(canvas, inputNodes, const Color(0xFF0EA5E9), 8, progress > 0 ? 1.0 : 0.0);
    _drawNodes(canvas, hidden1Nodes, const Color(0xFF4F46E5), 9, phase1);
    _drawNodes(canvas, hidden2Nodes, const Color(0xFF7C3AED), 10, phase2);
    _drawOutputNodes(canvas, outputNodes, phase3);
  }

  List<Offset> _buildColumn(
    Size size,
    double x,
    int count, {
    required double topPadding,
    required double bottomPadding,
  }) {
    if (count == 1) {
      return <Offset>[Offset(x, size.height / 2)];
    }

    final double usableHeight = size.height - topPadding - bottomPadding;
    final double gap = usableHeight / (count - 1);
    return List<Offset>.generate(
      count,
      (int index) => Offset(x, topPadding + (gap * index)),
    );
  }

  void _drawConnections(
    Canvas canvas,
    List<Offset> from,
    List<Offset> to,
    double phase,
    Paint paint,
  ) {
    final int total = from.length * to.length;
    final int visible = (total * phase).floor();
    int drawn = 0;

    for (final Offset start in from) {
      for (final Offset end in to) {
        if (drawn >= visible) {
          return;
        }
        canvas.drawLine(start, end, paint);
        drawn++;
      }
    }
  }

  void _drawNodes(Canvas canvas, List<Offset> nodes, Color color, double radius, double phase) {
    final Paint fill = Paint()..color = color.withValues(alpha: 0.28 + (0.52 * phase));
    final Paint stroke = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;

    for (final Offset node in nodes) {
      canvas.drawCircle(node, radius, fill);
      canvas.drawCircle(node, radius, stroke);
    }
  }

  void _drawOutputNodes(Canvas canvas, List<Offset> nodes, double phase) {
    final Paint normalFill = Paint()..color = const Color(0x3360A5FA);
    final Paint normalStroke = Paint()
      ..color = const Color(0xFF60A5FA)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;
    final Paint activeFill = Paint()..color = const Color(0xAA34D399);
    final Paint activeStroke = Paint()
      ..color = const Color(0xFF34D399)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.5;

    for (int i = 0; i < nodes.length; i++) {
      final Offset node = nodes[i];
      final bool active = i == highlightedDigit && phase > 0.8;
      canvas.drawCircle(node, 11, active ? activeFill : normalFill);
      canvas.drawCircle(node, 11, active ? activeStroke : normalStroke);

      final TextPainter text = TextPainter(
        text: TextSpan(
          text: '$i',
          style: const TextStyle(color: Color(0xFF0F172A), fontWeight: FontWeight.w700),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      text.paint(canvas, Offset(node.dx - (text.width / 2), node.dy - (text.height / 2)));
    }
  }

  @override
  bool shouldRepaint(covariant NetworkDiagramPainter oldDelegate) {
    return oldDelegate.progress != progress || oldDelegate.highlightedDigit != highlightedDigit;
  }
}

class DigitPainter extends CustomPainter {
  const DigitPainter({
    required this.strokes,
    required this.logicalCanvasSize,
    super.repaint,
  });

  final List<List<Offset>> strokes;
  final double logicalCanvasSize;

  @override
  void paint(Canvas canvas, Size size) {
    final Paint background = Paint()..color = Colors.white;
    canvas.drawRect(Offset.zero & size, background);

    final Paint grid = Paint()
      ..color = const Color(0xFFF1F5F9)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1;

    for (double offset = 0; offset <= size.width; offset += size.width / 4) {
      canvas.drawLine(Offset(offset, 0), Offset(offset, size.height), grid);
      canvas.drawLine(Offset(0, offset), Offset(size.width, offset), grid);
    }

    final double scale = size.width / logicalCanvasSize;
    final Paint paint = Paint()
      ..color = Colors.black
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round
      ..strokeWidth = 18 * scale;

    for (final List<Offset> stroke in strokes) {
      if (stroke.isEmpty) {
        continue;
      }

      if (stroke.length == 1) {
        final Offset point = Offset(stroke.first.dx * scale, stroke.first.dy * scale);
        canvas.drawCircle(point, 9 * scale, Paint()..color = Colors.black);
        continue;
      }

      final Path path = Path()
        ..moveTo(stroke.first.dx * scale, stroke.first.dy * scale);
      for (int i = 1; i < stroke.length; i++) {
        path.lineTo(stroke[i].dx * scale, stroke[i].dy * scale);
      }
      canvas.drawPath(path, paint);
    }
  }

  @override
  bool shouldRepaint(covariant DigitPainter oldDelegate) => true;
}

















