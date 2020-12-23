

PImage cuda1img;
int R = 2;
int S = 10;

void setup() {
  size(200, 400);
  background(255);
  
  cuda1img = loadImage("out.jpg");
  cuda1img.resize(width / R, height / R);
}

void draw() {
  //background(255);
  saveFrame("frames/" + nf(frameCount, 5) + ".tif");
  cuda1();
}

void cuda1() {
  scale(R);
  image(cuda1img, 0, 0);
  
  noStroke();
  fill(0, 255, 0);
  int x = frameCount % (width / R / S);
  int y = frameCount / ((width / R / S));
  rect(x * S, y * S, S, S);
}
