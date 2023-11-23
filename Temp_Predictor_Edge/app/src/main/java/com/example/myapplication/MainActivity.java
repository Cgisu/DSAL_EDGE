package com.example.myapplication;

import android.content.res.AssetFileDescriptor;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

import org.tensorflow.lite.Interpreter;
public class MainActivity extends AppCompatActivity {

    TextView tv;
    Button btn;
    EditText et;

    Interpreter tflite;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        tv = findViewById(R.id.textView);
        btn = findViewById(R.id.button);
        et = findViewById(R.id.editTextNumber);

        try {
            tflite = new Interpreter(loadModelFile());
        } catch (Exception ex){
            ex.printStackTrace();
        }


        btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                float prediction = DoInference(et.getText().toString());
                tv.setText(Float.toString(prediction));
            }
        });
    }

    private float DoInference(String input) {
        float[] inputVal = new float[1];
        inputVal[0] = Float.valueOf(input);
        float[][] output = new float[1][1];
        tflite.run(inputVal, output);
        float inferedValue = output[0][0];
        return inferedValue;
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("degree.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declareLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declareLength);
    }


}