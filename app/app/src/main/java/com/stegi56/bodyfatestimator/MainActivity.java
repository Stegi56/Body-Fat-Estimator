package com.stegi56.bodyfatestimator;

import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetFileDescriptor;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.widget.EditText;
import android.widget.TextView;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.os.Bundle;
import org.tensorflow.lite.Interpreter;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

import org.tensorflow.lite.Interpreter;


public class MainActivity extends AppCompatActivity {
    private EditText ankleField;
    private EditText kneeField;
    private EditText thighField;
    private EditText bicepField;
    private EditText waistField;
    private EditText forearmField;
    private EditText hipsField;
    private EditText heightField4;
    private EditText chestField;
    private EditText neckField;
    private EditText ageField;
    private EditText heightField;
    private EditText weightField;
    private TextView percentFat;

    private Interpreter tflite;
    private float[] inputValues;
    private float output;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        ankleField = findViewById(R.id.ankleField);
        kneeField = findViewById(R.id.kneeField);
        thighField = findViewById(R.id.thighField);
        bicepField = findViewById(R.id.bicepField);
        waistField = findViewById(R.id.waistField);
        forearmField = findViewById(R.id.forearmField);
        hipsField = findViewById(R.id.hipsField);
        heightField4 = findViewById(R.id.heightField4);
        chestField = findViewById(R.id.chestField);
        neckField = findViewById(R.id.neckField);
        ageField = findViewById(R.id.ageField);
        heightField = findViewById(R.id.heightField);
        weightField = findViewById(R.id.weightField);
        percentFat = findViewById(R.id.percentFat);

        ankleField.addTextChangedListener(textWatcher);
        kneeField.addTextChangedListener(textWatcher);
        thighField.addTextChangedListener(textWatcher);
        bicepField.addTextChangedListener(textWatcher);
        waistField.addTextChangedListener(textWatcher);
        forearmField.addTextChangedListener(textWatcher);
        hipsField.addTextChangedListener(textWatcher);
        heightField4.addTextChangedListener(textWatcher);
        chestField.addTextChangedListener(textWatcher);
        neckField.addTextChangedListener(textWatcher);
        ageField.addTextChangedListener(textWatcher);
        heightField.addTextChangedListener(textWatcher);
        weightField.addTextChangedListener(textWatcher);

        try {
            tflite = new Interpreter(loadModelFile(getAssets(), "model.tflite"));
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    private MappedByteBuffer loadModelFile(AssetManager assets, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelPath);
        FileInputStream fileInputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = fileInputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private TextWatcher textWatcher = new TextWatcher() {
        @Override
        public void beforeTextChanged(CharSequence charSequence, int start, int count, int after) {
        }

        @Override
        public void onTextChanged(CharSequence charSequence, int start, int before, int count) {
            updatePercentFat();
        }

        @Override
        public void afterTextChanged(Editable editable) {
        }
    };

    private void updatePercentFat () {
        // kg to lbs + cm to inch
        //(weight * weight) / height
        inputValues[0] = (((Float.parseFloat(weightField.getText().toString()) / 2.2f) *
                            (Float.parseFloat(weightField.getText().toString()) / 2.2f))
                        / (Float.parseFloat(heightField.getText().toString()) * 0.3937f);
        inputValues[1] = Float.parseFloat(kneeField.getText().toString());
        // ... and so on for other input fields

        // Perform inference using the TensorFlow Lite model
        tflite.run(inputValues, outputValues);

        // Update the percentFat TextView with the output prediction
        float estimatedFatPercentage = outputValues[0];
        percentFat.setText(String.format("%.2f%%", estimatedFatPercentage));

    }
}