//!MAGPIE EFFECT
//!VERSION 2
//!OUTPUT_WIDTH INPUT_WIDTH
//!OUTPUT_HEIGHT INPUT_HEIGHT

//!TEXTURE
Texture2D INPUT;

//!SAMPLER
//!FILTER LINEAR
SamplerState sam;

//!COMMON
static const float PI = 3.14159265f;
#define RELU(x) max(x, 0)
#define TANH(x) tanh(x)
#define SIGMOID(x) 1 / (1 + exp(-(x)))
#define SILU(x) (x) * SIGMOID(x)
#define ELU(x) max(0, x) + min(0, exp(x) - 1)
#define SELU(x) 1.05 * (max(0, x) + min(0, 1.67 * (exp(x) - 1)))
#define GELU2(x) 0.5 * (x) * (1 + tanh(0.7978845608 * min(10, max((x) * (1 + 0.044715 * (x) * (x)), -10))))
#define GELU(x) (x) * SIGMOID(1.702 * (x))
#define READ(value, pos, scale) (value) * step(0, (pos).x) * step(0, (pos).y) * step((pos).x, __inputSize.x * scale) * step((pos).y, __inputSize.y * scale)

const static float3x3 rgb2yuv = {
    0.299, 0.587, 0.114,
   -0.169, -0.331, 0.5,
   0.5, -0.419, -0.081
};

const static float3x3 yuv2rgb = {
	1, -0.00093, 1.401687,
	1, -0.3437, -0.71417,
	1, 1.77216, 0.00099
};


//!BEGIN
float load(int2 pos) {
    float3 rgb = INPUT[pos].rgb;
    float3 yuv = mul(rgb2yuv, rgb);
	return yuv.x;
}
//!BEGIN

//!END
void write(float value, int2 pos, float scale) {
    float3 rgb = INPUT.SampleLevel(sam, (pos + 0.5f) * 1.0 / (scale * GetInputSize()), 0).rgb;
    float3 yuv = mul(rgb2yuv, rgb);
    yuv.x = value;
    rgb = mul(yuv2rgb, yuv);
    WriteToOutput(pos, rgb);
}
//!END