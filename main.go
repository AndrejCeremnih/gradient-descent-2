package main

import (
	"encoding/csv"
	"fmt"
	"image"
	"image/color"
	"log"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgimg"
)

func Plot(ps ...plot.Plotter) *image.RGBA {
	p := plot.New()
	p.Add(append([]plot.Plotter{
		plotter.NewGrid(),
	}, ps...)...)
	img := image.NewRGBA(image.Rect(0, 0, 640, 480))
	c := vgimg.NewWith(vgimg.UseImage(img))
	p.Draw(draw.New(c))
	return c.Image().(*image.RGBA)
}

func main() {
	rand := rand.New(rand.NewSource(time.Now().UnixNano()))
	ebiten.SetWindowSize(640, 480)
	ebiten.SetWindowTitle("Gradient descent")

	const (
		epochs              = 2000
		printEveryNthEpochs = 100
		learningRateW       = 0.5e-3
		learningRateB       = 0.7

		plotLoss = false // Loss curve: true, Resulting line: false.

		inputPoints                      = 25
		inputPointsMinX, inputPointsMaxX = 5, 20
		inputPointsRandY                 = 1 // Makes sure Ys aren't on the line, but around it. Randomly.
		startValueRange                  = 1 // Start values for weights are in range [-startValueRange, startValueRange].

	)

	houses, err := readFromCSV("data/house_prices.csv")
	if err != nil {
		log.Fatalf("Can't read Houses from CSV: %v", err)
	}
	var inputs, labels []float64
	types, colors := make([][]float64, len(houses)), make([][]float64, len(houses))
	typeId := map[string][]float64{
		"Duplex":        []float64{1, 0, 0, 0, 0},
		"Detached":      []float64{0, 1, 0, 0, 0},
		"Semi-detached": []float64{0, 0, 1, 0, 0},
		"Townhouse":     []float64{0, 0, 0, 1, 0},
		"Multi-family":  []float64{0, 0, 0, 0, 1},
	}
	colorId := map[string]int{
		"brown":  0,
		"yellow": 1,
		"white":  2,
		"blue":   3,
		"green":  4,
	}
	for i, house := range houses {
		labels = append(labels, house.Price)
		inputs = append(inputs, house.Square)

		colorIndex := colorId[house.WallColor]
		colors[i] = make([]float64, 5)
		colors[i][colorIndex] = 1
	}

	xys := make([]plotter.XYs, 5)
	for i := 0; i < len(inputs); i++ {
		for j := 0; j < 5; j++ {
			if types[i][j] == 1 {
				xys[j] = append(xys[j], plotter.XY{X: inputs[i], Y: labels[i]})
			}
		}
	}
	var inputsScatter []*plotter.Scatter
	color := []color.RGBA{
		{0, 0, 0, 255},
		{255, 0, 0, 255},
		{0, 255, 0, 255},
		{0, 0, 255, 255},
		{255, 0, 255, 255},
		{0, 255, 255, 255},
	}

	for i := 0; i < 5; i++ {
		tmp, _ := plotter.NewScatter(xys[i])
		inputsScatter = append(inputsScatter, tmp)
		inputsScatter[i].Color = color[i]
	}

	img := make(chan *image.RGBA, 1) // Have at most one image in the channel.
	render := func(x *image.RGBA) {
		select {
		case <-img: // Drain the channel.
			img <- x // Put the new image in.
		case img <- x: // Or just put the new image in.
		}
	}
	go func() {
		w := make([]float64, 12)
		for i := range w {
			w[i] = startValueRange - rand.Float64()*2*startValueRange
		}
		var loss plotter.XYs
		for i := 0; i < epochs; i++ {
			y := inference(inputs, w, types)
			loss = append(loss, plotter.XY{
				X: float64(i),
				Y: msl(labels, y),
			})
			lossLines, _ := plotter.NewLine(loss)
			if plotLoss {
				render(Plot(lossLines))
			} else {
				const extra = (inputPointsMaxX - inputPointsMinX) / 10
				xs := []float64{inputPointsMinX - extra, inputPointsMaxX + extra}
				ys := inference(xs, w, types)
				resLine, _ := plotter.NewLine(plotter.XYs{{X: xs[0], Y: ys[0]}, {X: xs[1], Y: ys[1]}})
				render(Plot(inputsScatter, resLine))
			}
			dw, db := dmsl(inputs, labels, y)
			// w += dw * learningRateW
			// b += db * learningRateB
			//time.Sleep(30 * time.Millisecond)
			if i%printEveryNthEpochs == 0 {
				fmt.Printf(`Epoch #%d
	loss: %.4f
	dw: %.4f, db: %.4f
	w : %.4f,  b: %.4f
`, i, loss[len(loss)-1].Y, dw, db, w)
			}
		}
		fmt.Println(w)
	}()

	if err := ebiten.RunGame(&App{Img: img}); err != nil {
		log.Fatal(err)
	}
}

func inference(inputs, w []float64, t [][]float64) (res []float64) {
	for i, x := range inputs {
		res = append(res, (w[0]*t[i][0]+w[1]*t[i][1]+w[2]*t[i][2]+w[3]*t[i][3]+w[4]*t[i][4]+w[5])+(w[6]*t[i][0]+w[7]*t[i][1]+w[8]*t[i][2]+w[9]*t[i][3]+w[10]*t[i][4]+w[11])*x)
	}
	return res
}

func msl(labels, y []float64) (loss float64) {
	for i := range labels {
		loss += (labels[i] - y[i]) * (labels[i] - y[i])
	}
	return loss / float64(len(labels))
}

func dmsl(inputs, labels, y []float64) (dw, db float64) {
	for i := range labels {
		diff := labels[i] - y[i]
		dw += inputs[i] * diff
		db += diff
	}
	return 2 * dw / float64(len(labels)), 2 * db / float64(len(labels))
}

type House struct {
	Square    float64
	Type      string
	Price     float64
	WallColor string
}

func readFromCSV(filename string) ([]House, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	reader := csv.NewReader(file)
	reader.FieldsPerRecord = 4
	data, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	var houses []House
	for i, record := range data {
		if i == 0 {
			continue
		}
		square, err := strconv.ParseFloat(record[0], 64)
		if err != nil {
			log.Fatalf("Invalid square: %v", record)
			continue
		}

		price, err := strconv.ParseFloat(record[2], 64)
		if err != nil {
			log.Fatalf("Invalid price: %v", record)
			continue
		}

		houses = append(houses, House{
			Square:    square,
			Type:      record[1],
			Price:     price,
			WallColor: record[3],
		})
	}
	return houses, nil
}
