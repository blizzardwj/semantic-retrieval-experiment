"""
语义匹配评估脚本

该脚本用于评估不同语义匹配模型的性能，计算AUC指标并找出最佳阈值。
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class SemanticMatchingEvaluator:
    """
    语义匹配评估器，用于计算不同模型的AUC指标和最佳阈值。
    
    主要功能:
    - 加载实验结果数据
    - 计算每个模型的ROC曲线和AUC值
    - 找出每个模型的最佳阈值
    - 可视化ROC曲线和PR曲线
    - 进行阈值敏感性分析
    """
    
    def __init__(self, data_path: str, output_dir: Optional[str] = None):
        """
        初始化评估器。
        
        Args:
            data_path: 实验结果数据文件路径
            output_dir: 输出目录，用于保存图表和结果
        """
        self.data_path = data_path
        self.output_dir = output_dir or os.path.dirname(data_path)
        
        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 模型列表
        self.models = [
            'word2vec_match_res',  # Word2Vec模型匹配结果
            'bge_large_match_res', # BGE Large模型匹配结果
            'bge_m3_match_res'     # BGE M3模型匹配结果
        ]
        
        # 模型显示名称
        self.model_display_names = {
            'word2vec_match_res': 'Word2Vec',
            'bge_large_match_res': 'BGE-Large',
            'bge_m3_match_res': 'BGE-M3'
        }
        
        # 加载数据
        self.data = None
        self._load_data()
    
    def _load_data(self):
        """加载实验结果数据，过滤出有效行"""
        print(f"加载数据: {self.data_path}")
        
        # 读取CSV文件
        self.data = pd.read_csv(self.data_path)
        
        # 过滤出有效行（包含所有模型结果的行）
        valid_rows = self.data.dropna(subset=self.models)
        
        # 过滤出标签为0或1的行
        valid_rows = valid_rows[valid_rows['label'].isin([0, 1])]
        
        # 将标签转换为整数类型
        valid_rows['label'] = valid_rows['label'].astype(int)
        
        # 更新数据
        self.data = valid_rows
        
        print(f"有效数据行数: {len(self.data)}")
        print(f"标签分布: \n{self.data['label'].value_counts()}")
    
    def calculate_auc(self, model_name: str) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        计算指定模型的AUC值和ROC曲线。
        
        Args:
            model_name: 模型名称
            
        Returns:
            Tuple包含:
            - AUC值
            - FPR数组
            - TPR数组
            - 阈值数组
            - 最佳阈值
        """
        if model_name not in self.models:
            raise ValueError(f"未知模型: {model_name}")
        
        # 获取标签和预测分数
        y_true = self.data['label'].values
        y_score = self.data[model_name].values
        
        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        
        # 计算AUC值
        roc_auc = auc(fpr, tpr)
        
        # 计算最佳阈值 (Youden's J statistic)
        j_scores = tpr - fpr
        best_threshold_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_threshold_idx]
        
        return roc_auc, fpr, tpr, thresholds, best_threshold
    
    def evaluate_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        评估所有模型，计算AUC和最佳阈值。
        
        Returns:
            包含每个模型评估结果的字典
        """
        results = {}
        
        for model in self.models:
            print(f"评估模型: {self.model_display_names[model]}")
            
            # 计算AUC和ROC曲线
            roc_auc, fpr, tpr, thresholds, best_threshold = self.calculate_auc(model)
            
            # 计算PR曲线
            y_true = self.data['label'].values
            y_score = self.data[model].values
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
            avg_precision = average_precision_score(y_true, y_score)
            
            # 使用最佳阈值计算准确率
            y_pred = (y_score >= best_threshold).astype(int)
            accuracy = np.mean(y_pred == y_true)
            
            # 计算混淆矩阵
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity = tn / (tn + fp)
            sensitivity = tp / (tp + fn)
            f1_score = 2 * tp / (2 * tp + fp + fn)
            
            # 保存结果
            results[model] = {
                'auc': roc_auc,
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds,
                'best_threshold': best_threshold,
                'precision': precision,
                'recall': recall,
                'pr_thresholds': pr_thresholds,
                'avg_precision': avg_precision,
                'accuracy': accuracy,
                'specificity': specificity,
                'sensitivity': sensitivity,
                'f1_score': f1_score
            }
            
            print(f"  AUC: {roc_auc:.4f}")
            print(f"  最佳阈值: {best_threshold:.4f}")
            print(f"  准确率 (使用最佳阈值): {accuracy:.4f}")
            print(f"  特异性: {specificity:.4f}")
            print(f"  敏感性: {sensitivity:.4f}")
            print(f"  F1分数: {f1_score:.4f}")
            print(f"  平均精确率 (PR曲线下面积): {avg_precision:.4f}")
        
        return results
    
    def plot_roc_curves(self, results: Dict[str, Dict[str, Any]], save_path: Optional[str] = None):
        """
        绘制所有模型的ROC曲线。
        
        Args:
            results: 模型评估结果
            save_path: 保存路径，如果为None则显示图表
        """
        plt.figure(figsize=(10, 8))
        
        for model in self.models:
            model_results = results[model]
            plt.plot(
                model_results['fpr'], 
                model_results['tpr'], 
                label=f"{self.model_display_names[model]} (AUC = {model_results['auc']:.4f})"
            )
        
        # 绘制随机猜测的基准线
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC曲线已保存到: {save_path}")
        else:
            plt.show()
    
    def plot_pr_curves(self, results: Dict[str, Dict[str, Any]], save_path: Optional[str] = None):
        """
        绘制所有模型的PR曲线。
        
        Args:
            results: 模型评估结果
            save_path: 保存路径，如果为None则显示图表
        """
        plt.figure(figsize=(10, 8))
        
        for model in self.models:
            model_results = results[model]
            plt.plot(
                model_results['recall'], 
                model_results['precision'], 
                label=f"{self.model_display_names[model]} (AP = {model_results['avg_precision']:.4f})"
            )
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="best")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PR曲线已保存到: {save_path}")
        else:
            plt.show()
    
    def perform_threshold_sensitivity_analysis(self, model_name: str) -> pd.DataFrame:
        """
        对指定模型进行阈值敏感性分析。
        
        Args:
            model_name: 模型名称
            
        Returns:
            包含不同阈值下性能指标的DataFrame
        """
        if model_name not in self.models:
            raise ValueError(f"未知模型: {model_name}")
        
        # 获取标签和预测分数
        y_true = self.data['label'].values
        y_score = self.data[model_name].values
        
        # 生成一系列阈值
        thresholds = np.linspace(0.1, 0.95, 18)
        
        # 计算每个阈值下的性能指标
        results = []
        for threshold in thresholds:
            y_pred = (y_score >= threshold).astype(int)
            accuracy = np.mean(y_pred == y_true)
            
            # 计算混淆矩阵
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity = tn / (tn + fp)
            sensitivity = tp / (tp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1_score = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            
            results.append({
                'Threshold': threshold,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': sensitivity,
                'Specificity': specificity,
                'F1 Score': f1_score
            })
        
        return pd.DataFrame(results)
    
    def plot_threshold_sensitivity(self, model_name: str, sensitivity_df: pd.DataFrame, save_path: Optional[str] = None):
        """
        绘制阈值敏感性分析图表。
        
        Args:
            model_name: 模型名称
            sensitivity_df: 敏感性分析结果DataFrame
            save_path: 保存路径，如果为None则显示图表
        """
        plt.figure(figsize=(12, 8))
        
        # 绘制各指标随阈值变化的曲线
        plt.plot(sensitivity_df['Threshold'], sensitivity_df['Accuracy'], 'o-', label='Accuracy')
        plt.plot(sensitivity_df['Threshold'], sensitivity_df['Precision'], 's-', label='Precision')
        plt.plot(sensitivity_df['Threshold'], sensitivity_df['Recall'], '^-', label='Recall/Sensitivity')
        plt.plot(sensitivity_df['Threshold'], sensitivity_df['Specificity'], 'd-', label='Specificity')
        plt.plot(sensitivity_df['Threshold'], sensitivity_df['F1 Score'], '*-', label='F1 Score')
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title(f'Threshold Sensitivity Analysis - {self.model_display_names[model_name]}')
        plt.legend(loc="best")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"阈值敏感性分析图已保存到: {save_path}")
        else:
            plt.show()
    
    def save_results_to_csv(self, results: Dict[str, Dict[str, Any]], save_path: str):
        """
        将评估结果保存到CSV文件。
        
        Args:
            results: 模型评估结果
            save_path: 保存路径
        """
        # 创建结果DataFrame
        result_data = []
        
        for model in self.models:
            model_results = results[model]
            result_data.append({
                'Model': self.model_display_names[model],
                'AUC': model_results['auc'],
                'Best Threshold': model_results['best_threshold'],
                'Accuracy (at Best Threshold)': model_results['accuracy'],
                'Specificity': model_results['specificity'],
                'Sensitivity/Recall': model_results['sensitivity'],
                'F1 Score': model_results['f1_score'],
                'Average Precision': model_results['avg_precision']
            })
        
        result_df = pd.DataFrame(result_data)
        
        # 保存到CSV
        result_df.to_csv(save_path, index=False)
        print(f"评估结果已保存到: {save_path}")
    
    def run_evaluation(self):
        """运行完整的评估流程"""
        print("开始评估语义匹配模型...")
        
        # 评估所有模型
        results = self.evaluate_all_models()
        
        # 绘制ROC曲线
        roc_save_path = os.path.join(self.output_dir, 'roc_curves.png')
        self.plot_roc_curves(results, roc_save_path)
        
        # 绘制PR曲线
        pr_save_path = os.path.join(self.output_dir, 'pr_curves.png')
        self.plot_pr_curves(results, pr_save_path)
        
        # 对每个模型进行阈值敏感性分析
        for model in self.models:
            print(f"\n对模型 {self.model_display_names[model]} 进行阈值敏感性分析...")
            sensitivity_df = self.perform_threshold_sensitivity_analysis(model)
            
            # 保存敏感性分析结果
            sensitivity_save_path = os.path.join(self.output_dir, f'{model}_threshold_sensitivity.csv')
            sensitivity_df.to_csv(sensitivity_save_path, index=False)
            print(f"敏感性分析结果已保存到: {sensitivity_save_path}")
            
            # 绘制敏感性分析图表
            plot_save_path = os.path.join(self.output_dir, f'{model}_threshold_sensitivity.png')
            self.plot_threshold_sensitivity(model, sensitivity_df, plot_save_path)
        
        # 保存结果到CSV
        results_save_path = os.path.join(self.output_dir, 'evaluation_results.csv')
        self.save_results_to_csv(results, results_save_path)
        
        print("评估完成!")
        
        # 打印最终结果
        print("\n最终评估结果:")
        for model in self.models:
            model_results = results[model]
            print(f"{self.model_display_names[model]}:")
            print(f"  AUC: {model_results['auc']:.4f}")
            print(f"  最佳阈值: {model_results['best_threshold']:.4f}")
            print(f"  准确率 (使用最佳阈值): {model_results['accuracy']:.4f}")
            print(f"  特异性: {model_results['specificity']:.4f}")
            print(f"  敏感性/召回率: {model_results['sensitivity']:.4f}")
            print(f"  F1分数: {model_results['f1_score']:.4f}")
            print(f"  平均精确率: {model_results['avg_precision']:.4f}")
            print("")


def main():
    """主函数，解析命令行参数并运行评估"""
    parser = argparse.ArgumentParser(description="语义匹配评估脚本")
    parser.add_argument("--data-path", type=str, default="data/dataset1_exp_1742352405.csv", 
                        help="实验结果数据文件路径")
    parser.add_argument("--output-dir", type=str, default=None, 
                        help="输出目录，用于保存图表和结果，默认为数据文件所在目录")
    
    args = parser.parse_args()
    
    # 初始化评估器
    evaluator = SemanticMatchingEvaluator(
        data_path=args.data_path,
        output_dir=args.output_dir
    )
    
    # 运行评估
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
