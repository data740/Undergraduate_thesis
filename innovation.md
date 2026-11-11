创新点1：在world model训练阶段对Z~t~输入进行混合输入，以P%的比例（0.3左右）混入Z~t~^predict^

原因：STORM架构中WORLD MODEL的训练阶段只预测下一帧，输入的都是上一帧的正确答案；在Actor-Critic训练时，世界模型想象未来几帧通常会传递误差，在训练阶段也没有做过相关训练

创新点2：在world model训练阶段根据h~t~除了预测Z~t+1~,还要预测Z~t+2~等后面K帧

原因：完善监督信号

创新点3：世界模型采样，一部分按照均匀采样，另一部分按照wm loss高的采样

创新点4：在world model训练时，根据h~t~输出K个先验头，只前向传递，不传递到到wm主干；在Critic-Actor训练时，K个先验头的分布差异作为额外奖励加和到Actor的环境奖励，使Actor走wm不擅长的而且对得分有用区域，Replay_buffer会更加完善，提高wm的性能（simulus）



